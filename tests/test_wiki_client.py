import json
from pathlib import Path

import pytest

from backend import wiki_client


# ---------------------------------------------------------
# Helpers para resetear estado global en cada test
# ---------------------------------------------------------


def _reset_cache(monkeypatch) -> None:
    """Resetea el estado interno de wiki_client para que los tests sean aislados."""
    monkeypatch.setattr(wiki_client, "_wiki_cache", {}, raising=False)
    monkeypatch.setattr(wiki_client, "_wiki_cache_loaded", False, raising=False)


# ---------------------------------------------------------
# _normalize_title
# ---------------------------------------------------------


def test_normalize_title_basic() -> None:
    assert wiki_client._normalize_title("  La   Película ") == "la película"
    assert wiki_client._normalize_title("") == ""
    assert wiki_client._normalize_title("  ") == ""
    assert wiki_client._normalize_title("TÍTULO RARO!!") == "título raro!!".lower()


# ---------------------------------------------------------
# _load_wiki_cache / _save_wiki_cache
# ---------------------------------------------------------


def test_load_wiki_cache_nonexistent_file(tmp_path: Path, monkeypatch) -> None:
    _reset_cache(monkeypatch)
    cache_path = tmp_path / "wiki_cache.json"
    # Forzamos a que wiki_client use esta ruta
    monkeypatch.setattr(wiki_client, "WIKI_CACHE_PATH", cache_path, raising=False)

    # Aún no existe el fichero → cache vacía
    wiki_client._load_wiki_cache()
    assert wiki_client._wiki_cache == {}
    assert wiki_client._wiki_cache_loaded is True


def test_save_and_load_wiki_cache_roundtrip(tmp_path: Path, monkeypatch) -> None:
    _reset_cache(monkeypatch)
    cache_path = tmp_path / "wiki_cache.json"
    monkeypatch.setattr(wiki_client, "WIKI_CACHE_PATH", cache_path, raising=False)

    # Simulamos que ya está cargado y con contenido
    monkeypatch.setattr(wiki_client, "_wiki_cache_loaded", True, raising=False)
    original = {
        "imdb:tt123": {"Title": "Sample", "imdbID": "tt123"},
        "title:2000:sample": {"Title": "Sample", "Year": "2000"},
    }
    monkeypatch.setattr(wiki_client, "_wiki_cache", dict(original), raising=False)

    # Guardar
    wiki_client._save_wiki_cache()
    assert cache_path.exists()

    # Reset y recargar
    monkeypatch.setattr(wiki_client, "_wiki_cache_loaded", False, raising=False)
    monkeypatch.setattr(wiki_client, "_wiki_cache", {}, raising=False)
    wiki_client._load_wiki_cache()

    assert wiki_client._wiki_cache == original
    assert wiki_client._wiki_cache_loaded is True


# ---------------------------------------------------------
# get_movie_record: HIT en caché sin reintento
# ---------------------------------------------------------


def test_get_movie_record_cache_hit_no_retry(monkeypatch) -> None:
    _reset_cache(monkeypatch)

    # Preparar cache con una entrada
    record = {"Title": "Cached Movie", "imdbID": "tt0000001"}
    cache = {"imdb:tt0000001": record}
    monkeypatch.setattr(wiki_client, "_wiki_cache", cache, raising=False)
    monkeypatch.setattr(wiki_client, "_wiki_cache_loaded", True, raising=False)

    # OMDB_RETRY_EMPTY_CACHE = False → no se reintenta ni toca OMDb
    monkeypatch.setattr(wiki_client, "OMDB_RETRY_EMPTY_CACHE", False, raising=False)

    # Si se llamara a search_omdb_by_imdb_id, queremos enterarnos
    def fake_search_omdb_by_imdb_id(imdb_id: str):
        raise AssertionError("No debería llamarse search_omdb_by_imdb_id en cache HIT sin retry")

    monkeypatch.setattr(
        wiki_client, "search_omdb_by_imdb_id", fake_search_omdb_by_imdb_id, raising=False
    )

    # También evitamos escrituras en disco
    monkeypatch.setattr(wiki_client, "_save_wiki_cache", lambda: None, raising=False)

    result = wiki_client.get_movie_record(
        title="Cached Movie",
        year=2000,
        imdb_id_hint="tt0000001",
    )

    assert result is record
    assert result["imdbID"] == "tt0000001"


# ---------------------------------------------------------
# get_movie_record: HIT con OMDB_RETRY_EMPTY_CACHE=True y refresh OMDb
# ---------------------------------------------------------


def test_get_movie_record_cache_hit_with_retry(monkeypatch) -> None:
    _reset_cache(monkeypatch)

    # Registro en caché pero sin ratings → forzamos is_omdb_data_empty_for_ratings=True
    record = {"Title": "Old Movie", "imdbID": "tt0000002"}
    cache = {"imdb:tt0000002": record}
    monkeypatch.setattr(wiki_client, "_wiki_cache", cache, raising=False)
    monkeypatch.setattr(wiki_client, "_wiki_cache_loaded", True, raising=False)

    monkeypatch.setattr(wiki_client, "OMDB_RETRY_EMPTY_CACHE", True, raising=False)

    # is_omdb_data_empty_for_ratings siempre True para este test
    monkeypatch.setattr(
        wiki_client,
        "is_omdb_data_empty_for_ratings",
        lambda rec: True,
        raising=False,
    )

    calls: list[str] = []

    def fake_search_omdb_by_imdb_id(imdb_id: str):
        calls.append(imdb_id)
        return {
            "Response": "True",
            "Title": "Refreshed Movie",
            "imdbID": imdb_id,
            "imdbRating": "7.5",
        }

    monkeypatch.setattr(
        wiki_client, "search_omdb_by_imdb_id", fake_search_omdb_by_imdb_id, raising=False
    )
    monkeypatch.setattr(wiki_client, "_save_wiki_cache", lambda: None, raising=False)

    result = wiki_client.get_movie_record(
        title="Old Movie",
        year=1990,
        imdb_id_hint="tt0000002",
    )

    # Se ha llamado a OMDb
    assert calls == ["tt0000002"]
    # Se devuelve el registro refrescado
    assert result is not None
    assert result["Title"] == "Refreshed Movie"
    assert result["imdbID"] == "tt0000002"
    # Caché actualizada en memoria
    assert wiki_client._wiki_cache["imdb:tt0000002"]["Title"] == "Refreshed Movie"


# ---------------------------------------------------------
# get_movie_record: MISS con imdb_id_hint y sin Wikidata
# ---------------------------------------------------------


def test_get_movie_record_cache_miss_with_imdb_hint_no_wikidata(monkeypatch) -> None:
    _reset_cache(monkeypatch)

    monkeypatch.setattr(wiki_client, "_wiki_cache_loaded", True, raising=False)
    monkeypatch.setattr(wiki_client, "_wiki_cache", {}, raising=False)
    monkeypatch.setattr(wiki_client, "OMDB_RETRY_EMPTY_CACHE", False, raising=False)

    # Forzamos que Wikidata no devuelva nada aunque se llamara
    monkeypatch.setattr(
        wiki_client, "_wikidata_search_by_imdb", lambda imdb_id: None, raising=False
    )

    calls: list[tuple[str, dict]] = []

    def fake_search_omdb_by_imdb_id(imdb_id: str):
        calls.append((imdb_id, {}))
        return {
            "Response": "True",
            "Title": "OMDb Title",
            "imdbID": imdb_id,
        }

    monkeypatch.setattr(
        wiki_client, "search_omdb_by_imdb_id", fake_search_omdb_by_imdb_id, raising=False
    )
    monkeypatch.setattr(wiki_client, "_save_wiki_cache", lambda: None, raising=False)

    result = wiki_client.get_movie_record(
        title="Some Movie",
        year=2010,
        imdb_id_hint="tt9999999",
    )

    # Se debe haber llamado a OMDb por ID
    assert calls and calls[0][0] == "tt9999999"
    assert result is not None
    assert result["Title"] == "OMDb Title"
    assert result["imdbID"] == "tt9999999"


# ---------------------------------------------------------
# get_movie_record: MISS sin imdb_id_hint → search_omdb_with_candidates
# ---------------------------------------------------------


def test_get_movie_record_cache_miss_without_imdb_hint_uses_candidates(monkeypatch) -> None:
    _reset_cache(monkeypatch)

    monkeypatch.setattr(wiki_client, "_wiki_cache_loaded", True, raising=False)
    monkeypatch.setattr(wiki_client, "_wiki_cache", {}, raising=False)
    monkeypatch.setattr(wiki_client, "OMDB_RETRY_EMPTY_CACHE", False, raising=False)

    # Forzamos que Wikidata no encuentre nada por título
    monkeypatch.setattr(
        wiki_client, "_wikidata_search_by_title", lambda *a, **k: None, raising=False
    )

    calls: list[tuple[str, int | None]] = []

    def fake_search_omdb_with_candidates(title: str, year: int | None):
        calls.append((title, year))
        return {
            "Response": "True",
            "Title": f"Candidate for {title}",
            "Year": str(year) if year is not None else None,
            "imdbID": "ttcand0001",
        }

    monkeypatch.setattr(
        wiki_client,
        "search_omdb_with_candidates",
        fake_search_omdb_with_candidates,
        raising=False,
    )
    monkeypatch.setattr(wiki_client, "_save_wiki_cache", lambda: None, raising=False)

    result = wiki_client.get_movie_record(
        title="Candidate Movie",
        year=1999,
        imdb_id_hint=None,
    )

    # Confirmamos que se ha ido por la ruta de candidatos
    assert calls == [("Candidate Movie", 1999)]
    assert result is not None
    assert result["Title"] == "Candidate for Candidate Movie"
    assert result["imdbID"] == "ttcand0001"