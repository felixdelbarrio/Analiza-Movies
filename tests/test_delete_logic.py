# tests/test_delete_logic.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import pytest

from backend import delete_logic


def make_rows_from_dicts(dicts: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    return list(dicts)


# -------------------------------------------------------------------
# Casos básicos con lista de dicts
# -------------------------------------------------------------------


def test_delete_files_skip_when_not_exists(tmp_path: Path) -> None:
    missing = tmp_path / "no_exists.mkv"
    rows = make_rows_from_dicts(
        [
            {"title": "Missing", "file": str(missing)},
        ]
    )

    ok, err, logs = delete_logic.delete_files_from_rows(rows, delete_dry_run=False)

    assert ok == 0
    assert err == 0
    assert any("archivo no existe" in msg for msg in logs)
    # Obviamente sigue sin existir
    assert not missing.exists()


def test_delete_files_skip_when_not_file(tmp_path: Path) -> None:
    # Creamos un directorio en lugar de un archivo
    directory = tmp_path / "some_dir"
    directory.mkdir()

    rows = make_rows_from_dicts(
        [
            {"title": "Not a file", "file": str(directory)},
        ]
    )

    ok, err, logs = delete_logic.delete_files_from_rows(rows, delete_dry_run=False)

    assert ok == 0
    assert err == 0
    assert any("no es un fichero" in msg for msg in logs)
    # El directorio sigue existiendo
    assert directory.exists()
    assert directory.is_dir()


def test_delete_files_dry_run(tmp_path: Path) -> None:
    f = tmp_path / "movie.mkv"
    f.write_text("data", encoding="utf-8")

    rows = make_rows_from_dicts(
        [
            {"title": "DryRun", "file": str(f)},
        ]
    )

    ok, err, logs = delete_logic.delete_files_from_rows(rows, delete_dry_run=True)

    assert ok == 0
    assert err == 0
    # No se borra el fichero
    assert f.exists()
    assert any("[DRY RUN]" in msg for msg in logs)


def test_delete_files_real_delete_ok(tmp_path: Path) -> None:
    f = tmp_path / "to_delete.mkv"
    f.write_text("data", encoding="utf-8")

    rows = make_rows_from_dicts(
        [
            {"title": "DeleteMe", "file": str(f)},
        ]
    )

    ok, err, logs = delete_logic.delete_files_from_rows(rows, delete_dry_run=False)

    assert ok == 1
    assert err == 0
    assert any("[OK] BORRADO" in msg for msg in logs)
    assert not f.exists()


def test_delete_files_real_delete_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Simula un error en unlink() para comprobar que se contabiliza como error
    y se loguea correctamente.
    """
    f = tmp_path / "cant_delete.mkv"
    f.write_text("data", encoding="utf-8")

    # Guardamos el original para usarlo en rutas que no nos interesen
    original_unlink = Path.unlink

    def fake_unlink(self: Path, *args, **kwargs) -> None:  # noqa: ARG001
        if self == f:
            raise OSError("simulated unlink failure")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr("pathlib.Path.unlink", fake_unlink)

    rows = make_rows_from_dicts(
        [
            {"title": "ErrorFile", "file": str(f)},
        ]
    )

    ok, err, logs = delete_logic.delete_files_from_rows(rows, delete_dry_run=False)

    assert ok == 0
    assert err == 1
    assert any("[ERROR] ErrorFile" in msg for msg in logs)
    # El archivo sigue existiendo porque falló el unlink
    assert f.exists()


# -------------------------------------------------------------------
# Casos adicionales: filas sin file / title, y DataFrame
# -------------------------------------------------------------------


def test_delete_files_skip_when_file_missing_key(tmp_path: Path) -> None:
    f = tmp_path / "ignored.mkv"
    f.write_text("data", encoding="utf-8")

    rows = make_rows_from_dicts(
        [
            {"title": "NoFileField"},  # sin clave "file"
        ]
    )

    ok, err, logs = delete_logic.delete_files_from_rows(rows, delete_dry_run=False)

    assert ok == 0
    assert err == 0
    assert any("sin ruta de archivo" in msg for msg in logs)
    # El archivo que creamos para el test no se ha tocado (ni siquiera se referencia)
    assert f.exists()


def test_delete_files_works_with_dataframe(tmp_path: Path) -> None:
    f1 = tmp_path / "df1.mkv"
    f2 = tmp_path / "df2.mkv"
    f1.write_text("1", encoding="utf-8")
    f2.write_text("2", encoding="utf-8")

    df = pd.DataFrame(
        [
            {"title": "DF1", "file": str(f1)},
            {"title": "DF2", "file": str(f2)},
        ]
    )

    ok, err, logs = delete_logic.delete_files_from_rows(df, delete_dry_run=False)

    assert ok == 2
    assert err == 0
    assert not f1.exists()
    assert not f2.exists()
    assert any("DF1" in msg for msg in logs)
    assert any("DF2" in msg for msg in logs)


def test_delete_files_invalid_rows_type_raises() -> None:
    # rows no iterable ni DataFrame -> debe lanzar TypeError
    with pytest.raises(TypeError):
        delete_logic.delete_files_from_rows(rows=123, delete_dry_run=True)  # type: ignore[arg-type]