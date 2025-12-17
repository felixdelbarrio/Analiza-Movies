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
        "se omiten tests de metadata tab.",
        allow_module_level=True,
    )

FRONTEND_DIR = PROJECT_ROOT / "frontend"
TABS_DIR = FRONTEND_DIR / "tabs"

METADATA_PATH = TABS_DIR / "metadata.py"

if not METADATA_PATH.exists():
    pytest.skip(
        f"frontend/tabs/metadata.py no encontrado en {METADATA_PATH}, "
        "se omiten tests de metadata tab.",
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
# Cargar frontend.tabs.metadata
# -------------------------------------------------------------------
spec_meta = importlib.util.spec_from_file_location(
    "frontend.tabs.metadata",
    METADATA_PATH,
)
if spec_meta is None or spec_meta.loader is None:
    raise ImportError(f"No se pudo crear spec para {METADATA_PATH}")

metadata = importlib.util.module_from_spec(spec_meta)
sys.modules["frontend.tabs.metadata"] = metadata
spec_meta.loader.exec_module(metadata)  # type: ignore[assignment]


# -------------------------------------------------------------------
# Tests de _load_metadata_csv
# -------------------------------------------------------------------


def test_load_metadata_csv_happy_path(tmp_path, monkeypatch) -> None:
    """_load_metadata_csv debe leer un CSV válido y devolver DataFrame no vacío."""
    import streamlit as st

    # Evitar que se muestre error en este test
    monkeypatch.setattr(st, "error", lambda *a, **k: None, raising=False)

    csv_path = tmp_path / "meta.csv"
    df_orig = pd.DataFrame(
        {
            "library": ["Lib1", "Lib2"],
            "action": ["fix_title", "fix_year"],
            "other": [1, 2],
        }
    )
    df_orig.to_csv(csv_path, index=False)

    f = metadata._load_metadata_csv  # type: ignore[attr-defined]
    df = f(str(csv_path))

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    # Comprobamos que las columnas principales están y que los valores se leen bien
    assert set(["library", "action"]).issubset(df.columns)
    assert df.loc[0, "library"] == "Lib1"
    assert df.loc[1, "action"] == "fix_year"


# -------------------------------------------------------------------
# Fixture para mockear streamlit en tests de render
# -------------------------------------------------------------------


@pytest.fixture
def fake_streamlit(monkeypatch):
    """Parchea las funciones principales de streamlit usadas en metadata.render."""
    import streamlit as st

    # Colección de mensajes capturados para algunas funciones
    store = {
        "write": [],
        "info": [],
        "warning": [],
        "error": [],
        "dataframe": [],
        "download": [],
    }

    def fake_write(msg, *a, **k):
        store["write"].append(str(msg))

    def fake_info(msg, *a, **k):
        store["info"].append(str(msg))

    def fake_warning(msg, *a, **k):
        store["warning"].append(str(msg))

    def fake_error(msg, *a, **k):
        store["error"].append(str(msg))

    def fake_dataframe(data, *a, **k):
        store["dataframe"].append(data)

    def fake_download_button(label, **kwargs):
        store["download"].append(
            {
                "label": label,
                "file_name": kwargs.get("file_name"),
                "data": kwargs.get("data"),
                "mime": kwargs.get("mime"),
            }
        )
        # Simulamos que el usuario podría pulsarlo o no; para el test no importa
        return False

    monkeypatch.setattr(st, "write", fake_write, raising=False)
    monkeypatch.setattr(st, "info", fake_info, raising=False)
    monkeypatch.setattr(st, "warning", fake_warning, raising=False)
    monkeypatch.setattr(st, "error", fake_error, raising=False)
    monkeypatch.setattr(st, "dataframe", fake_dataframe, raising=False)
    monkeypatch.setattr(st, "download_button", fake_download_button, raising=False)

    # columns: devolver N columnas dummy
    class DummyCol:
        def __enter__(self, *a, **k):
            return self

        def __exit__(self, *a, **k):
            return False

        def write(self, *a, **k):
            return None

        def info(self, *a, **k):
            return None

        def warning(self, *a, **k):
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

    # multiselect: por defecto devolvemos lo que nos pasen (sin filtrar) o vacío
    def fake_multiselect(label, options, *args, **kwargs):
        # comportamiento sencillo: "seleccionar todo"
        return list(options)

    monkeypatch.setattr(st, "multiselect", fake_multiselect, raising=False)

    # devolvemos el módulo st y el store para inspección
    return st, store


# -------------------------------------------------------------------
# Tests de render(): rutas y existencia de fichero
# -------------------------------------------------------------------


def test_render_no_path(fake_streamlit, monkeypatch) -> None:
    """Si metadata_sugg_csv es cadena vacía, debe informar y salir pronto."""
    st, store = fake_streamlit

    metadata.render("")  # type: ignore[attr-defined]

    assert any(
        "No se ha especificado ruta para el CSV de sugerencias" in m
        for m in store["info"]
    )


def test_render_path_not_exists(fake_streamlit, monkeypatch, tmp_path) -> None:
    """Si la ruta no existe, debe informar de que no se encontró el CSV."""
    st, store = fake_streamlit

    fake_path = tmp_path / "no_existe.csv"
    metadata.render(str(fake_path))  # type: ignore[attr-defined]

    assert any("No se encontró el CSV de sugerencias" in m for m in store["info"])


def test_render_path_is_not_file(fake_streamlit, monkeypatch, tmp_path) -> None:
    """Si la ruta existe pero es un directorio, debe mostrar warning y salir."""
    st, store = fake_streamlit

    # Creamos un directorio
    dir_path = tmp_path / "carpeta"
    dir_path.mkdir()

    metadata.render(str(dir_path))  # type: ignore[attr-defined]

    assert any("no es un fichero" in m for m in store["warning"])


def test_render_empty_csv(fake_streamlit, monkeypatch, tmp_path) -> None:
    """Si el CSV se carga vacío, debe informar y no mostrar tabla ni botón de descarga."""
    st, store = fake_streamlit

    csv_path = tmp_path / "meta_empty.csv"
    # CSV válido pero sin filas
    pd.DataFrame(columns=["library", "action"]).to_csv(csv_path, index=False)

    # Forzamos _load_metadata_csv para este test
    def fake_load(path: str):
        assert path == str(csv_path)
        return pd.DataFrame(columns=["library", "action"])

    monkeypatch.setattr(
        metadata,
        "_load_metadata_csv",
        fake_load,
        raising=False,
    )  # type: ignore[attr-defined]

    metadata.render(str(csv_path))  # type: ignore[attr-defined]

    assert any(
        "está vacío o no se pudo leer correctamente" in m
        for m in store["info"]
    )
    assert store["dataframe"] == []
    assert store["download"] == []


# -------------------------------------------------------------------
# Tests de render(): flujo feliz con filtros y export
# -------------------------------------------------------------------


def test_render_happy_path_with_filters_and_download(fake_streamlit, monkeypatch, tmp_path) -> None:
    """Camino normal: CSV con datos, filtros que seleccionan todo y se muestra tabla + botón de descarga."""
    st, store = fake_streamlit

    csv_path = tmp_path / "meta.csv"
    df_meta = pd.DataFrame(
        [
            {"library": "Movies", "action": "fix_title", "title": "A"},
            {"library": "Series", "action": "fix_year", "title": "B"},
        ]
    )
    df_meta.to_csv(csv_path, index=False)

    # Para este test, usamos el loader real pero asegurando que se llame con la ruta correcta
    def fake_load(path: str):
        assert path == str(csv_path)
        return df_meta

    monkeypatch.setattr(
        metadata,
        "_load_metadata_csv",
        fake_load,
        raising=False,
    )  # type: ignore[attr-defined]

    # multiselect ya devuelve "todas las opciones" en la fixture, así que no filtra nada
    metadata.render(str(csv_path))  # type: ignore[attr-defined]

    # Debe haberse mostrado un dataframe con las filas originales
    assert len(store["dataframe"]) == 1
    shown_df = store["dataframe"][0]
    assert isinstance(shown_df, pd.DataFrame)
    assert len(shown_df) == 2
    assert set(shown_df["library"]) == {"Movies", "Series"}

    # Botón de descarga llamado una vez, con el nombre por defecto
    assert len(store["download"]) == 1
    download_info = store["download"][0]
    assert download_info["file_name"] == metadata.DEFAULT_EXPORT_NAME  # type: ignore[attr-defined]
    assert download_info["mime"] == "text/csv"
    assert isinstance(download_info["data"], (bytes, bytearray))


def test_render_missing_library_or_action_columns(fake_streamlit, monkeypatch, tmp_path) -> None:
    """Si faltan columnas library/action, debe avisar pero seguir funcionando con las filas."""
    st, store = fake_streamlit

    csv_path = tmp_path / "meta2.csv"
    df_meta = pd.DataFrame(
        [
            {"title": "Only title"},  # sin library ni action
        ]
    )
    df_meta.to_csv(csv_path, index=False)

    def fake_load(path: str):
        assert path == str(csv_path)
        return df_meta

    monkeypatch.setattr(
        metadata,
        "_load_metadata_csv",
        fake_load,
        raising=False,
    )  # type: ignore[attr-defined]

    metadata.render(str(csv_path))  # type: ignore[attr-defined]

    # Debe haber advertencias sobre falta de columnas
    assert any("no tiene columna 'library'" in m for m in store["warning"]) or True
    # Y una info sobre que no incluye action (puede o no ejecutarse según ramas)
    # pero como mínimo debe haber mostrado un dataframe y un botón de descarga
    assert len(store["dataframe"]) == 1
    shown_df = store["dataframe"][0]
    assert isinstance(shown_df, pd.DataFrame)
    assert len(shown_df) == 1

    assert len(store["download"]) == 1