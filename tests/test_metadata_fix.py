# tests/test_metadata_fix.py
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from backend import metadata_fix


# -------------------------------------------------------------------
# Helpers para crear "movie" de Plex simulado
# -------------------------------------------------------------------


def make_movie(
    title: Any = "Plex Title",
    year: Any = 2000,
    library: str = "MyLib",
    guid: str = "guid://123",
    with_save: bool = False,
) -> SimpleNamespace:
    """Crea un objeto movie simple para tests."""
    obj_dict: Dict[str, Any] = {
        "title": title,
        "year": year,
        "librarySectionTitle": library,
        "guid": guid,
    }
    if with_save:
        called = {"value": False}

        def _save() -> None:
            called["value"] = True

        obj_dict["save"] = _save
        obj_dict["_save_called"] = called  # para inspeccionar luego

    return SimpleNamespace(**obj_dict)


# -------------------------------------------------------------------
# Tests de normalización
# -------------------------------------------------------------------


def test_normalize_title_basic_cases() -> None:
    f = metadata_fix._normalize_title  # type: ignore[attr-defined]

    assert f("  El Título  ") == "el título"
    assert f("EL.TÍTULO!!") == "el título"
    assert f("") is None
    assert f(None) is None  # type: ignore[arg-type]


def test_normalize_title_removes_extra_spaces_and_punctuation() -> None:
    f = metadata_fix._normalize_title  # type: ignore[attr-defined]
    assert f("  La, peli...  de   prueba!!! ") == "la peli de prueba"


def test_normalize_year_basic_and_invalid() -> None:
    f = metadata_fix._normalize_year  # type: ignore[attr-defined]

    assert f(2020) == 2020
    assert f("1999") == 1999
    assert f("1999  ") == 1999
    assert f(None) is None
    assert f("no-year") is None
    # floats no se consideran años válidos según la implementación actual
    assert f(3.14) is None


# -------------------------------------------------------------------
# Tests de generate_metadata_suggestions_row
# -------------------------------------------------------------------


def test_generate_metadata_suggestions_row_no_omdb_returns_none() -> None:
    movie = make_movie()
    result = metadata_fix.generate_metadata_suggestions_row(movie, None)
    assert result is None


def test_generate_metadata_suggestions_row_no_diff_returns_none() -> None:
    movie = make_movie(title="Same Title", year=2000)
    omdb_data = {"Title": "Same Title", "Year": "2000"}

    result = metadata_fix.generate_metadata_suggestions_row(movie, omdb_data)
    assert result is None


def test_generate_metadata_suggestions_row_title_diff_only() -> None:
    movie = make_movie(title="Plex Title", year=2000)
    omdb_data = {"Title": "Other Title", "Year": "2000"}

    row = metadata_fix.generate_metadata_suggestions_row(movie, omdb_data)
    assert row is not None
    assert row["plex_title"] == "Plex Title"
    assert row["omdb_title"] == "Other Title"
    assert row["action"] == "Fix title"

    suggestions = json.loads(row["suggestions_json"])  # type: ignore[arg-type]
    assert suggestions["new_title"] == "Other Title"
    assert "new_year" not in suggestions


def test_generate_metadata_suggestions_row_year_diff_only() -> None:
    movie = make_movie(title="Same Title", year=1999)
    omdb_data = {"Title": "Same Title", "Year": "2001"}

    row = metadata_fix.generate_metadata_suggestions_row(movie, omdb_data)
    assert row is not None
    assert row["action"] == "Fix year"

    suggestions = json.loads(row["suggestions_json"])  # type: ignore[arg-type]
    # _normalize_year convierte a int
    assert suggestions["new_year"] == 2001
    assert "new_title" not in suggestions


def test_generate_metadata_suggestions_row_title_and_year_diff() -> None:
    movie = make_movie(title="Plex", year=1990)
    omdb_data = {"Title": "OMDb", "Year": "2000"}

    row = metadata_fix.generate_metadata_suggestions_row(movie, omdb_data)
    assert row is not None
    assert row["action"] == "Fix title & year"

    suggestions = json.loads(row["suggestions_json"])  # type: ignore[arg-type]
    assert suggestions["new_title"] == "OMDb"
    assert suggestions["new_year"] == 2000


def test_generate_metadata_suggestions_row_handles_non_string_title() -> None:
    movie = make_movie(title="Peli", year=2000)
    omdb_data = {"Title": 12345, "Year": "2000"}  # Title no es str

    row = metadata_fix.generate_metadata_suggestions_row(movie, omdb_data)
    # Title no se considera válido (no str) y, al normalizar, no hay diff
    assert row is None


# -------------------------------------------------------------------
# Tests de apply_metadata_suggestion
# -------------------------------------------------------------------


def test_apply_metadata_suggestion_no_suggestions_json() -> None:
    movie = make_movie()
    row = {
        "suggestions_json": "",
    }
    logs = metadata_fix.apply_metadata_suggestion(movie, row)
    # Debe registrar cabecera + mensaje de "no hay sugerencias"
    assert any("No hay sugerencias" in line for line in logs)


def test_apply_metadata_suggestion_only_logs_when_apply_false(monkeypatch) -> None:
    movie = make_movie()
    # Forzamos flags
    monkeypatch.setattr(metadata_fix, "METADATA_APPLY_CHANGES", False)
    monkeypatch.setattr(metadata_fix, "METADATA_DRY_RUN", True)

    suggestions = {"new_title": "New Title", "new_year": 2010}
    row = {"suggestions_json": json.dumps(suggestions)}

    logs = metadata_fix.apply_metadata_suggestion(movie, row)

    # No debe cambiar los atributos reales
    assert movie.title == "Plex Title"
    assert movie.year == 2000

    # Debe contener mensaje claro de solo log
    assert any("METADATA_APPLY_CHANGES=False" in line for line in logs)


def test_apply_metadata_suggestion_dry_run_true(monkeypatch) -> None:
    movie = make_movie()
    monkeypatch.setattr(metadata_fix, "METADATA_APPLY_CHANGES", True)
    monkeypatch.setattr(metadata_fix, "METADATA_DRY_RUN", True)

    suggestions = {"new_title": "Dry Run Title", "new_year": 2015}
    row = {"suggestions_json": json.dumps(suggestions)}

    logs = metadata_fix.apply_metadata_suggestion(movie, row)

    # Sin cambios reales en movie
    assert movie.title == "Plex Title"
    assert movie.year == 2000

    assert any("METADATA_DRY_RUN=True" in line for line in logs)


def test_apply_metadata_suggestion_real_changes_and_save_called(monkeypatch) -> None:
    movie = make_movie(with_save=True)
    monkeypatch.setattr(metadata_fix, "METADATA_APPLY_CHANGES", True)
    monkeypatch.setattr(metadata_fix, "METADATA_DRY_RUN", False)

    suggestions = {"new_title": "Real Title", "new_year": 2022}
    row = {"suggestions_json": json.dumps(suggestions)}

    logs = metadata_fix.apply_metadata_suggestion(movie, row)

    assert movie.title == "Real Title"
    assert movie.year == 2022
    assert any("Cambios aplicados en campos" in line for line in logs)

    # Comprobar que save() fue llamado
    assert getattr(movie, "_save_called")["value"] is True  # type: ignore[index]


def test_apply_metadata_suggestion_suggestions_json_as_dict(monkeypatch) -> None:
    movie = make_movie()
    monkeypatch.setattr(metadata_fix, "METADATA_APPLY_CHANGES", True)
    monkeypatch.setattr(metadata_fix, "METADATA_DRY_RUN", False)

    # suggestions_json ya es un dict (no string)
    suggestions = {"new_title": "Dict Title"}
    row = {"suggestions_json": suggestions}

    logs = metadata_fix.apply_metadata_suggestion(movie, row)

    assert movie.title == "Dict Title"
    assert any("Cambios aplicados en campos" in line for line in logs)


def test_apply_metadata_suggestion_handles_invalid_json(monkeypatch) -> None:
    movie = make_movie()
    monkeypatch.setattr(metadata_fix, "METADATA_APPLY_CHANGES", False)
    monkeypatch.setattr(metadata_fix, "METADATA_DRY_RUN", True)

    # JSON inválido → debe caer en sugerencias vacías
    row = {"suggestions_json": "{not-json"}
    logs = metadata_fix.apply_metadata_suggestion(movie, row)
    assert any("No hay sugerencias" in line for line in logs)


def test_apply_metadata_suggestion_respects_silent_mode(monkeypatch, capsys) -> None:
    """Con SILENT_MODE=True no debería imprimir ni loggear nada visible."""
    movie = make_movie()
    monkeypatch.setattr(metadata_fix, "METADATA_APPLY_CHANGES", False)
    monkeypatch.setattr(metadata_fix, "METADATA_DRY_RUN", True)
    monkeypatch.setattr(metadata_fix, "SILENT_MODE", True)

    suggestions = {"new_title": "New Silent Title"}
    row = {"suggestions_json": json.dumps(suggestions)}

    logs = metadata_fix.apply_metadata_suggestion(movie, row)

    # Las funciones internas devuelven logs en la lista, pero no deben imprimir por stdout
    captured = capsys.readouterr()
    assert captured.out == ""
    assert isinstance(logs, list)
    assert len(logs) >= 1