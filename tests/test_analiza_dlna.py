from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Tuple

import os
import json
import pytest

from backend import analiza_dlna


# ============================================================
# Helpers
# ============================================================


class Recorder:
    """Helper para registrar llamadas a funciones mockeadas."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))


# ============================================================
# Tests de helpers internos (_is_video_file, _guess_title_year)
# ============================================================


def test_is_video_file_true_and_false(tmp_path: Path) -> None:
    video = tmp_path / "movie.mkv"
    text = tmp_path / "notes.txt"

    video.write_bytes(b"123")
    text.write_text("hola", encoding="utf-8")

    assert analiza_dlna._is_video_file(video) is True
    assert analiza_dlna._is_video_file(text) is False
    # Directorio tampoco debe considerarse vídeo
    subdir = tmp_path / "dir"
    subdir.mkdir()
    assert analiza_dlna._is_video_file(subdir) is False


@pytest.mark.parametrize(
    "filename,expected_title,expected_year",
    [
        ("Good.Movie (2001).mkv", "Good.Movie", 2001),
        ("Another.Movie.1999.1080p.mkv", "Another.Movie.1999.1080p", 1999),
        ("TitleWithoutYear.mkv", "TitleWithoutYear", None),
    ],
)
def test_guess_title_year_patterns(
    tmp_path: Path,
    filename: str,
    expected_title: str,
    expected_year: int | None,
) -> None:
    f = tmp_path / filename
    f.write_bytes(b"data")
    title, year = analiza_dlna._guess_title_year(f)
    assert title == expected_title
    assert year == expected_year


# ============================================================
# Tests del flujo principal analyze_dlna_server
# ============================================================


def _make_fake_files(tmp_path: Path) -> list[Path]:
    """
    Crea un pequeño set de ficheros de vídeo de prueba.

    - Good.Movie (2001).mkv   → título 'Good.Movie', year=2001
    - Bad.Movie.2005.mkv      → título 'Bad.Movie.2005', year=2005 (patrón .YYYY.)
    """
    good = tmp_path / "Good.Movie (2001).mkv"
    bad = tmp_path / "Bad.Movie.2005.mkv"

    good.write_bytes(b"123456")
    bad.write_bytes(b"abcdef")

    return [good, bad]


def test_analyze_dlna_server_with_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 1) _ask_root_directory devuelve nuestro tmp_path
    monkeypatch.setattr(analiza_dlna, "_ask_root_directory", lambda: tmp_path)

    # 2) Forzamos la lista de vídeo a un set controlado
    files = _make_fake_files(tmp_path)
    monkeypatch.setattr(analiza_dlna, "_iter_video_files", lambda root: files)

    # 3) Stub de OMDb/Wiki: get_movie_record → datos mínimos
    def fake_get_movie_record(title: str, year: int | None, imdb_id_hint: str | None = None):
        # Devolvemos algo tipo OMDb + __wiki para comprobar enriquecimiento
        return {
            "Title": title,
            "Year": str(year) if year is not None else None,
            "Poster": f"http://poster/{title}",
            "Website": f"http://trailer/{title}",
            "imdbID": "tt1234567",
            "__wiki": {
                "wikidata_id": "Q1",
                "wikipedia_title": f"Wiki {title}",
            },
        }

    monkeypatch.setattr(analiza_dlna, "get_movie_record", fake_get_movie_record)

    # 4) Stub del core genérico analyze_input_movie
    collected_inputs: list[Tuple[str, int | None, str]] = []

    def fake_analyze_input_movie(dlna_input, fetch_omdb):
        # Registramos lo que entra para asegurar que _guess_title_year
        # se aplicó correctamente.
        collected_inputs.append(
            (dlna_input.title, dlna_input.year, dlna_input.file_path)
        )

        # Simulamos una decisión: GOOD → KEEP, BAD → DELETE
        decision = "KEEP" if "Good" in dlna_input.title else "DELETE"

        return {
            "source": dlna_input.source,
            "library": dlna_input.library,
            "title": dlna_input.title,
            "year": dlna_input.year,
            "imdb_rating": 7.0,
            "rt_score": 80,
            "imdb_votes": 1000,
            "plex_rating": None,
            "decision": decision,
            "reason": "reason",
            "misidentified_hint": "",
            "file": dlna_input.file_path,
            "file_size_bytes": dlna_input.file_size_bytes,
        }

    monkeypatch.setattr(analiza_dlna, "analyze_input_movie", fake_analyze_input_movie)

    # 5) sort_filtered_rows → identidad, pero registramos la llamada
    sort_rec = Recorder()

    def fake_sort_filtered_rows(rows):
        sort_rec(rows)
        return rows

    monkeypatch.setattr(analiza_dlna, "sort_filtered_rows", fake_sort_filtered_rows)

    # 6) Capturamos escrituras CSV
    rec_all = Recorder()
    rec_filtered = Recorder()
    rec_sugg = Recorder()

    monkeypatch.setattr(analiza_dlna, "write_all_csv", rec_all)
    monkeypatch.setattr(analiza_dlna, "write_filtered_csv", rec_filtered)
    monkeypatch.setattr(analiza_dlna, "write_suggestions_csv", rec_sugg)

    # 7) Prefijos
    monkeypatch.setattr(analiza_dlna, "OUTPUT_PREFIX", "OUTDLNA")
    monkeypatch.setattr(analiza_dlna, "METADATA_OUTPUT_PREFIX", "METADLNA")

    # 8) Capturamos logs info para no depender de stdout
    info_logs: list[str] = []
    monkeypatch.setattr(
        analiza_dlna._logger,
        "info",
        lambda msg, **kw: info_logs.append(str(msg)),
    )
    monkeypatch.setattr(
        analiza_dlna._logger,
        "warning",
        lambda msg, **kw: info_logs.append(str(msg)),
    )
    monkeypatch.setattr(
        analiza_dlna._logger,
        "error",
        lambda msg, **kw: info_logs.append(str(msg)),
    )

    # Ejecutamos el flujo principal
    analiza_dlna.analyze_dlna_server()

    # -----------------------------------------------------
    # Verificaciones
    # -----------------------------------------------------

    # 1) Entradas al core (títulos / años) coherentes
    collected_titles = {t for (t, y, p) in collected_inputs}
    collected_years = {y for (t, y, p) in collected_inputs}
    assert "Good.Movie" in collected_titles  # se parseó bien "Good.Movie (2001)"
    assert 2001 in collected_years
    # El otro título puede incluir ".2005" porque _guess_title_year no lo recorta
    assert any("Bad.Movie" in t for (t, _, _) in collected_inputs)

    # 2) write_all_csv llamado correctamente
    assert len(rec_all.calls) == 1
    (args_all, kwargs_all) = rec_all.calls[0]
    assert kwargs_all == {}
    out_all, rows_all = args_all
    assert out_all == "OUTDLNA_dlna_all.csv"
    assert len(rows_all) == 2

    titles_all = {r["title"] for r in rows_all}
    assert "Good.Movie" in titles_all
    # Título del malo puede ser "Bad.Movie.2005" (según _guess_title_year)
    assert any("Bad.Movie" in t for t in titles_all)

    # 3) write_filtered_csv contiene solo DELETE (el "Bad")
    assert len(rec_filtered.calls) == 1
    (args_f, kwargs_f) = rec_filtered.calls[0]
    assert kwargs_f == {}
    out_f, rows_f = args_f
    assert out_f == "OUTDLNA_dlna_filtered.csv"
    assert {r["decision"] for r in rows_f} == {"DELETE"}

    # 4) write_suggestions_csv se llama con lista vacía (por el momento DLNA no genera sugerencias)
    assert len(rec_sugg.calls) == 1
    (args_s, kwargs_s) = rec_sugg.calls[0]
    assert kwargs_s == {}
    out_s, rows_s = args_s
    assert out_s == "METADLNA_dlna.csv"
    assert rows_s == []

    # 5) sort_filtered_rows se llamó con las filas DELETE/MAYBE
    assert len(sort_rec.calls) == 1
    (args_sort, _kwargs_sort) = sort_rec.calls[0]
    rows_in = args_sort[0]
    assert all(r["decision"] in {"DELETE", "MAYBE"} for r in rows_in)

    # 6) Log final de completado
    assert any("Análisis completado." in m for m in info_logs)


def test_analyze_dlna_server_analyze_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Si analyze_input_movie lanza error para un fichero, se loggea y se sigue con el resto."""
    monkeypatch.setattr(analiza_dlna, "_ask_root_directory", lambda: tmp_path)

    files = _make_fake_files(tmp_path)
    monkeypatch.setattr(analiza_dlna, "_iter_video_files", lambda root: files)

    # Primer fichero lanza error, segundo funciona
    calls: list[str] = []

    def fake_analyze_input_movie(dlna_input, fetch_omdb):
        calls.append(dlna_input.file_path)
        if "Good.Movie" in dlna_input.file_path:
            raise RuntimeError("Test error")
        return {
            "source": dlna_input.source,
            "library": dlna_input.library,
            "title": dlna_input.title,
            "year": dlna_input.year,
            "imdb_rating": 5.0,
            "rt_score": 40,
            "imdb_votes": 100,
            "plex_rating": None,
            "decision": "DELETE",
            "reason": "reason",
            "misidentified_hint": "",
            "file": dlna_input.file_path,
            "file_size_bytes": dlna_input.file_size_bytes,
        }

    monkeypatch.setattr(analiza_dlna, "analyze_input_movie", fake_analyze_input_movie)

    # get_movie_record stub mínimo
    monkeypatch.setattr(
        analiza_dlna,
        "get_movie_record",
        lambda *a, **k: {"Title": "X", "Year": "2000"},
    )

    # CSV recorders
    rec_all = Recorder()
    rec_filtered = Recorder()
    rec_sugg = Recorder()
    monkeypatch.setattr(analiza_dlna, "write_all_csv", rec_all)
    monkeypatch.setattr(analiza_dlna, "write_filtered_csv", rec_filtered)
    monkeypatch.setattr(analiza_dlna, "write_suggestions_csv", rec_sugg)

    # Prefijos
    monkeypatch.setattr(analiza_dlna, "OUTPUT_PREFIX", "OUTERR")
    monkeypatch.setattr(analiza_dlna, "METADATA_OUTPUT_PREFIX", "METAERR")

    # Logs
    info_logs: list[str] = []
    monkeypatch.setattr(
        analiza_dlna._logger,
        "info",
        lambda msg, **kw: info_logs.append(str(msg)),
    )
    monkeypatch.setattr(
        analiza_dlna._logger,
        "warning",
        lambda msg, **kw: info_logs.append(str(msg)),
    )
    monkeypatch.setattr(
        analiza_dlna._logger,
        "error",
        lambda msg, **kw: info_logs.append(str(msg)),
    )

    analiza_dlna.analyze_dlna_server()

    # Debe haberse intentado analizar ambos ficheros
    assert len(calls) == 2

    # Pero solo el segundo acaba en los CSV (el primero provocó excepción)
    assert len(rec_all.calls) == 1
    (args_all, _kw_all) = rec_all.calls[0]
    _, rows_all = args_all
    assert len(rows_all) == 1
    only_row = rows_all[0]
    # El título del segundo fichero depende de _guess_title_year,
    # que mantendrá "Bad.Movie.2005" como title.
    assert "Bad.Movie" in only_row["title"]

    # El filtrado DELETE/MAYBE también solo debe contener esa fila
    assert len(rec_filtered.calls) == 1
    (args_f, _kw_f) = rec_filtered.calls[0]
    _, rows_f = args_f
    assert len(rows_f) == 1
    assert rows_f[0]["decision"] == "DELETE"

    # Se ha logueado el error para el primer fichero
    assert any("Error analizando" in m for m in info_logs)
    # Y se loguea completado
    assert any("Análisis completado." in m for m in info_logs)


def test_analyze_dlna_server_no_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Si no hay ficheros de vídeo, no se escriben CSVs y se loguea el mensaje correspondiente."""
    monkeypatch.setattr(analiza_dlna, "_ask_root_directory", lambda: tmp_path)
    monkeypatch.setattr(analiza_dlna, "_iter_video_files", lambda root: [])

    rec_all = Recorder()
    rec_filtered = Recorder()
    rec_sugg = Recorder()
    monkeypatch.setattr(analiza_dlna, "write_all_csv", rec_all)
    monkeypatch.setattr(analiza_dlna, "write_filtered_csv", rec_filtered)
    monkeypatch.setattr(analiza_dlna, "write_suggestions_csv", rec_sugg)

    info_logs: list[str] = []
    monkeypatch.setattr(
        analiza_dlna._logger,
        "info",
        lambda msg, **kw: info_logs.append(str(msg)),
    )

    analiza_dlna.analyze_dlna_server()

    # No debe haberse intentado escribir ningún CSV
    assert rec_all.calls == []
    assert rec_filtered.calls == []
    assert rec_sugg.calls == []

    # Y se loguea el mensaje de "no se han encontrado ficheros"
    assert any("No se han encontrado ficheros de vídeo" in m for m in info_logs)