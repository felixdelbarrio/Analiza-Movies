# tests/test_report_loader.py
import sys
import types
import pytest
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------
# Crear módulo falso frontend.data_utils ANTES de importar report_loader
# ---------------------------------------------------------------------

if "frontend" not in sys.modules:
    frontend_module = types.ModuleType("frontend")
    sys.modules["frontend"] = frontend_module
else:
    frontend_module = sys.modules["frontend"]  # type: ignore[assignment]

data_utils_module = types.ModuleType("frontend.data_utils")


def _fake_add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df  # comportamiento neutro por defecto


data_utils_module.add_derived_columns = _fake_add_derived_columns  # type: ignore[attr-defined]
sys.modules["frontend.data_utils"] = data_utils_module

from backend import report_loader  # noqa: E402


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def make_csv(path: Path, rows: list[dict]) -> None:
    import csv
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# _clean_base_dataframe
# ---------------------------------------------------------------------

def test_clean_base_dataframe_removes_thumb() -> None:
    df = pd.DataFrame(
        {
            "title": ["A"],
            "thumb": ["REMOVE"],
            "other": [123],
        }
    )
    cleaned = report_loader._clean_base_dataframe(df)
    assert "thumb" not in cleaned.columns
    assert list(cleaned.columns) == ["title", "other"]


def test_clean_base_dataframe_no_thumb() -> None:
    df = pd.DataFrame({"title": ["A"], "year": [2020]})
    cleaned = report_loader._clean_base_dataframe(df)
    assert list(cleaned.columns) == ["title", "year"]


# ---------------------------------------------------------------------
# _cast_text_columns
# ---------------------------------------------------------------------

def test_cast_text_columns_forces_string_for_text_columns() -> None:
    df = pd.DataFrame(
        {
            "poster_url": [None, 123],
            "trailer_url": [5.6, None],
            "omdb_json": [{"x": 1}, None],
            "other": [1, 2],
        }
    )
    casted = report_loader._cast_text_columns(df)

    for col in report_loader.TEXT_COLUMNS:
        assert casted[col].dtype == object
        assert all(isinstance(v, str) for v in casted[col])

    # La columna "other" debe mantenerse igual
    assert list(casted["other"]) == [1, 2]


def test_cast_text_columns_missing_columns_ok() -> None:
    df = pd.DataFrame({"title": ["A"]})
    casted = report_loader._cast_text_columns(df)
    assert df.equals(casted)


# ---------------------------------------------------------------------
# load_reports — casos principales
# ---------------------------------------------------------------------

def test_load_reports_basic(tmp_path: Path, monkeypatch) -> None:
    # Fake add_derived_columns para ver si se llama
    calls = {"count": 0}

    def fake_add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
        calls["count"] += 1
        return df.assign(derived=1)

    monkeypatch.setattr(report_loader, "add_derived_columns", fake_add_derived_columns)

    all_rows = [
        {"title": "A", "poster_url": None, "trailer_url": None, "omdb_json": "{}"},
        {"title": "B", "poster_url": "x", "trailer_url": None, "omdb_json": "{}"},
    ]
    all_csv = tmp_path / "all.csv"
    make_csv(all_csv, all_rows)

    filtered_rows = [
        {"title": "A", "poster_url": None, "trailer_url": None, "omdb_json": "{}"},
    ]
    filtered_csv = tmp_path / "filtered.csv"
    make_csv(filtered_csv, filtered_rows)

    df_all, df_filtered = report_loader.load_reports(str(all_csv), str(filtered_csv))

    assert isinstance(df_all, pd.DataFrame)
    assert isinstance(df_filtered, pd.DataFrame)
    assert "derived" in df_all.columns
    assert calls["count"] >= 1

    # Debe haber casteo de columnas
    for col in report_loader.TEXT_COLUMNS:
        assert df_all[col].dtype == object


def test_load_reports_missing_filtered(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(report_loader, "add_derived_columns", lambda df: df)

    all_rows = [{"title": "X", "poster_url": "", "trailer_url": "", "omdb_json": "{}"}]
    all_csv = tmp_path / "all.csv"
    make_csv(all_csv, all_rows)

    df_all, df_filtered = report_loader.load_reports(str(all_csv), None)

    assert isinstance(df_all, pd.DataFrame)
    assert df_filtered is None


def test_load_reports_filtered_does_not_exist(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(report_loader, "add_derived_columns", lambda df: df)

    all_rows = [{"title": "X", "poster_url": "", "trailer_url": "", "omdb_json": "{}"}]
    all_csv = tmp_path / "all.csv"
    make_csv(all_csv, all_rows)

    df_all, df_filtered = report_loader.load_reports(str(all_csv), "no-such-file.csv")
    assert isinstance(df_all, pd.DataFrame)
    assert df_filtered is None


def test_load_reports_missing_all_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        report_loader.load_reports(str(tmp_path / "no.csv"), None)


def test_load_reports_cleans_thumb(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(report_loader, "add_derived_columns", lambda df: df)

    rows = [
        {"title": "A", "thumb": "REMOVE", "poster_url": "", "trailer_url": "", "omdb_json": "{}"},
    ]
    all_csv = tmp_path / "all.csv"
    make_csv(all_csv, rows)

    df_all, _ = report_loader.load_reports(str(all_csv), None)

    assert "thumb" not in df_all.columns
    assert df_all["title"].iloc[0] == "A"


def test_load_reports_casts_text_columns(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(report_loader, "add_derived_columns", lambda df: df)

    rows = [{
        "title": "A",
        "poster_url": None,
        "trailer_url": 123,
        "omdb_json": {"x": 1},
    }]
    all_csv = tmp_path / "all.csv"
    make_csv(all_csv, rows)

    df_all, _ = report_loader.load_reports(str(all_csv), None)

    for col in report_loader.TEXT_COLUMNS:
        assert isinstance(df_all[col].iloc[0], str)


def test_load_reports_adds_derived_columns_called(tmp_path: Path, monkeypatch) -> None:
    tracker = {"called": False}

    def fake_add(df):
        tracker["called"] = True
        return df.assign(ok=1)

    monkeypatch.setattr(report_loader, "add_derived_columns", fake_add)

    rows = [{"title": "X", "poster_url": "", "trailer_url": "", "omdb_json": "{}"}]
    all_csv = tmp_path / "all.csv"
    make_csv(all_csv, rows)

    df_all, _ = report_loader.load_reports(str(all_csv), None)

    assert tracker["called"] is True
    assert "ok" in df_all.columns


# ---------------------------------------------------------------------
# 🔧 TEST CORREGIDO: CSV filtrado “raro” → pandas devuelve DF vacío
# ---------------------------------------------------------------------

def test_load_reports_handles_bad_filtered_csv(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(report_loader, "add_derived_columns", lambda df: df)

    # CSV completo válido
    all_rows = [{"title": "A", "poster_url": "", "trailer_url": "", "omdb_json": "{}"}]
    all_csv = tmp_path / "all.csv"
    make_csv(all_csv, all_rows)

    # CSV filtrado sintácticamente válido pero semánticamente basura
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("this,is,not,valid,csv\n\n\n", encoding="utf-8")

    df_all, df_filtered = report_loader.load_reports(str(all_csv), str(bad_csv))

    # df_all siempre correcto
    assert isinstance(df_all, pd.DataFrame)

    # pandas interpreta el CSV como DataFrame vacío (cabecera sin filas)
    assert isinstance(df_filtered, pd.DataFrame)
    assert df_filtered.empty