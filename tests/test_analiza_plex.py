from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from backend import analiza_plex


class Recorder:
    """Helper para registrar llamadas a funciones mockeadas."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))


def make_movie(title: str, year: int = 2000) -> SimpleNamespace:
    return SimpleNamespace(
        librarySectionTitle="MyLib",
        title=title,
        year=year,
        rating=None,
        ratingKey=f"rk-{title}",
        guid=f"imdb://tt{year:07d}",
        thumb=None,
        media=[],
    )


class FakeLibrary:
    def __init__(self, title: str, movies: list[SimpleNamespace]) -> None:
        self.title = title
        self._movies = movies

    def search(self) -> list[SimpleNamespace]:
        return self._movies


# ============================================================
# Happy path
# ============================================================


def test_analyze_all_libraries_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # 1) Fake conexión y bibliotecas
    fake_plex = object()
    monkeypatch.setattr(analiza_plex, "connect_plex", lambda: fake_plex)

    movies = [
        make_movie("Keep Movie", 2010),
        make_movie("Delete Movie", 2011),
        make_movie("Maybe Movie", 2012),
    ]
    libs = [FakeLibrary("Movies", movies)]
    monkeypatch.setattr(analiza_plex, "get_libraries_to_analyze", lambda plex: libs)

    # 2) Fake analyze_single_movie:
    #    - Keep Movie   -> decision KEEP, meta_sugg presente
    #    - Delete Movie -> decision DELETE, sin meta_sugg
    #    - Maybe Movie  -> decision MAYBE, sin meta_sugg
    def fake_analyze_single_movie(movie: Any):
        title = movie.title
        logs = [f"log for {title}"]
        if "Keep" in title:
            row = {
                "title": title,
                "decision": "KEEP",
                "file": f"/tmp/{title}.mkv",
            }
            meta = {"plex_guid": movie.guid, "action": "Fix title"}
        elif "Delete" in title:
            row = {
                "title": title,
                "decision": "DELETE",
                "file": f"/tmp/{title}.mkv",
            }
            meta = None
        else:
            row = {
                "title": title,
                "decision": "MAYBE",
                "file": f"/tmp/{title}.mkv",
            }
            meta = None
        return row, meta, logs

    monkeypatch.setattr(analiza_plex, "analyze_single_movie", fake_analyze_single_movie)

    # 3) sort_filtered_rows -> identidad, pero registramos la llamada
    sort_rec = Recorder()

    def fake_sort_filtered_rows(rows):
        sort_rec(rows)
        return rows

    monkeypatch.setattr(analiza_plex, "sort_filtered_rows", fake_sort_filtered_rows)

    # 4) Capturamos escrituras de CSV
    rec_all = Recorder()
    rec_filtered = Recorder()
    rec_sugg = Recorder()
    monkeypatch.setattr(analiza_plex, "write_all_csv", rec_all)
    monkeypatch.setattr(analiza_plex, "write_filtered_csv", rec_filtered)
    monkeypatch.setattr(analiza_plex, "write_suggestions_csv", rec_sugg)

    # 5) Prefijos
    monkeypatch.setattr(analiza_plex, "OUTPUT_PREFIX", "OUT")
    monkeypatch.setattr(analiza_plex, "METADATA_OUTPUT_PREFIX", "META")

    # 6) Capturamos logs
    info_logs: list[str] = []
    monkeypatch.setattr(
        analiza_plex._logger,
        "info",
        lambda msg, **kw: info_logs.append(str(msg)),
    )

    # Ejecutar
    analiza_plex.analyze_all_libraries()

    # --------------------------------------------
    # Verificaciones
    # --------------------------------------------

    # sort_filtered_rows debe haber recibido solo DELETE/MAYBE
    assert len(sort_rec.calls) == 1
    (args_sort, _kwargs_sort) = sort_rec.calls[0]
    filtered_in = args_sort[0]
    decisions_in = {r["decision"] for r in filtered_in}
    assert decisions_in == {"DELETE", "MAYBE"}
    titles_in = {r["title"] for r in filtered_in}
    assert titles_in == {"Delete Movie", "Maybe Movie"}

    # write_all_csv: 3 filas (todas)
    assert len(rec_all.calls) == 1
    (args_all, kwargs_all) = rec_all.calls[0]
    assert kwargs_all == {}
    out_path_all, rows_all = args_all
    assert out_path_all == "OUT_plex_all.csv"
    assert {r["title"] for r in rows_all} == {
        "Keep Movie",
        "Delete Movie",
        "Maybe Movie",
    }

    # write_filtered_csv: DELETE + MAYBE
    assert len(rec_filtered.calls) == 1
    (args_filt, kwargs_filt) = rec_filtered.calls[0]
    assert kwargs_filt == {}
    out_path_filt, rows_filt = args_filt
    assert out_path_filt == "OUT_plex_filtered.csv"
    assert {r["decision"] for r in rows_filt} == {"DELETE", "MAYBE"}
    assert {r["title"] for r in rows_filt} == {"Delete Movie", "Maybe Movie"}

    # write_suggestions_csv: solo meta_sugg de Keep Movie
    assert len(rec_sugg.calls) == 1
    (args_sugg, kwargs_sugg) = rec_sugg.calls[0]
    assert kwargs_sugg == {}
    out_path_sugg, rows_sugg = args_sugg
    assert out_path_sugg == "META_plex.csv"
    assert len(rows_sugg) == 1
    assert rows_sugg[0]["action"] == "Fix title"
    assert "plex_guid" in rows_sugg[0]

    # Log final de completado
    assert any("[PLEX] Análisis completado." in m for m in info_logs)
    # Y se registró el log de "Analizando biblioteca Plex: ..."
    assert any("Analizando biblioteca Plex: Movies" in m for m in info_logs)


# ============================================================
# Caso sin bibliotecas
# ============================================================


def test_analyze_all_libraries_no_libraries(monkeypatch: pytest.MonkeyPatch) -> None:
    # connect_plex devuelve algo dummy
    monkeypatch.setattr(analiza_plex, "connect_plex", lambda: object())
    # Ninguna biblioteca
    monkeypatch.setattr(analiza_plex, "get_libraries_to_analyze", lambda plex: [])

    rec_all = Recorder()
    rec_filtered = Recorder()
    rec_sugg = Recorder()
    monkeypatch.setattr(analiza_plex, "write_all_csv", rec_all)
    monkeypatch.setattr(analiza_plex, "write_filtered_csv", rec_filtered)
    monkeypatch.setattr(analiza_plex, "write_suggestions_csv", rec_sugg)

    monkeypatch.setattr(analiza_plex, "OUTPUT_PREFIX", "OUT2")
    monkeypatch.setattr(analiza_plex, "METADATA_OUTPUT_PREFIX", "META2")

    info_logs: list[str] = []
    monkeypatch.setattr(
        analiza_plex._logger,
        "info",
        lambda msg, **kw: info_logs.append(str(msg)),
    )

    analiza_plex.analyze_all_libraries()

    # Debe seguir escribiendo CSVs, pero con listas vacías
    assert len(rec_all.calls) == 1
    (args_all, kwargs_all) = rec_all.calls[0]
    assert kwargs_all == {}
    path_all, rows_all = args_all
    assert path_all == "OUT2_plex_all.csv"
    assert rows_all == []

    assert len(rec_filtered.calls) == 1
    (args_filt, kwargs_filt) = rec_filtered.calls[0]
    assert kwargs_filt == {}
    path_filt, rows_filt = args_filt
    assert path_filt == "OUT2_plex_filtered.csv"
    assert rows_filt == []

    assert len(rec_sugg.calls) == 1
    (args_sugg, kwargs_sugg) = rec_sugg.calls[0]
    assert kwargs_sugg == {}
    path_sugg, rows_sugg = args_sugg
    assert path_sugg == "META2_plex.csv"
    assert rows_sugg == []

    # Log de completado sigue presente
    assert any("[PLEX] Análisis completado." in m for m in info_logs)