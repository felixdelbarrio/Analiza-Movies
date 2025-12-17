# tests/test_analyze_input_core.py
from __future__ import annotations

from typing import Mapping

import pytest

from backend.movie_input import MovieInput
from backend import analyze_input_core  # <-- nombre correcto del módulo


# -------------------------------------------------------------------
# Helper para crear entradas MovieInput
# -------------------------------------------------------------------
def make_input(
    *,
    source: str = "dlna",
    library: str = "MyLibrary",
    title: str = "Sample Movie",
    year: int | None = 2020,
    file_path: str = "/path/to/file.mkv",
    file_size_bytes: int | None = 123456,
    imdb_id_hint: str | None = "tt0000001",
) -> MovieInput:
    return MovieInput(
        source=source,
        library=library,
        title=title,
        year=year,
        file_path=file_path,
        file_size_bytes=file_size_bytes,
        imdb_id_hint=imdb_id_hint,
        plex_guid=None,
        rating_key=None,
        thumb_url=None,
    )


# -------------------------------------------------------------------
# 1) Flujo básico con OMDb válido
# -------------------------------------------------------------------
def test_analyze_input_movie_basic_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    movie = make_input(
        title="My DLNA Movie",
        year=2015,
        imdb_id_hint="tt1234567",
    )

    # OMDb fake
    def fake_fetch_omdb(title: str, year: int | None) -> Mapping[str, object]:
        assert title == "My DLNA Movie"
        assert year == 2015
        return {
            "Title": "My DLNA Movie",
            "Year": "2015",
            "imdbRating": "7.5",
            "imdbVotes": "1,000",
            "Ratings": [{"Source": "Rotten Tomatoes", "Value": "80%"}],
        }

    def fake_decide_action(
        *,
        imdb_rating,
        imdb_votes,
        rt_score,
        year,
        metacritic_score=None,
    ):
        assert imdb_rating == 7.5
        assert imdb_votes == 1000
        assert rt_score == 80
        return "KEEP", "reason-from-decide"

    def fake_detect_misidentified(**kwargs):
        return "hint-misid"

    monkeypatch.setattr(analyze_input_core, "decide_action", fake_decide_action)
    monkeypatch.setattr(analyze_input_core, "detect_misidentified", fake_detect_misidentified)

    row = analyze_input_core.analyze_input_movie(movie, fake_fetch_omdb)

    # Entradas base
    assert row["source"] == "dlna"
    assert row["library"] == "MyLibrary"
    assert row["title"] == "My DLNA Movie"
    assert row["year"] == 2015

    # Ratings derivados de OMDb
    assert row["imdb_rating"] == 7.5
    assert row["imdb_votes"] == 1000
    assert row["rt_score"] == 80

    # Decision y hint
    assert row["decision"] == "KEEP"
    assert row["reason"] == "reason-from-decide"
    assert row["misidentified_hint"] == "hint-misid"

    # File
    assert row["file"] == "/path/to/file.mkv"
    assert row["file_size_bytes"] == 123456

    # imdb_id_hint transmitido
    assert row["imdb_id_hint"] == "tt1234567"


# -------------------------------------------------------------------
# 2) fetch_omdb lanza → se usa dict vacío
# -------------------------------------------------------------------
def test_analyze_input_movie_fetch_omdb_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    movie = make_input(title="Broken", year=2010)

    def fake_fetch_omdb_raise(title: str, year: int | None):
        raise RuntimeError("boom")

    def fake_extract_ratings(omdb):
        assert omdb == {} or omdb == {}
        return None, None, None

    def fake_decide_action(**kwargs):
        return "UNKNOWN", "no OMDb data"

    def fake_detect_misidentified(**kwargs):
        return ""

    monkeypatch.setattr(analyze_input_core, "extract_ratings_from_omdb", fake_extract_ratings)
    monkeypatch.setattr(analyze_input_core, "decide_action", fake_decide_action)
    monkeypatch.setattr(analyze_input_core, "detect_misidentified", fake_detect_misidentified)

    row = analyze_input_core.analyze_input_movie(movie, fake_fetch_omdb_raise)

    assert row["decision"] == "UNKNOWN"
    assert row["reason"] == "no OMDb data"
    assert row["imdb_rating"] is None
    assert row["misidentified_hint"] == ""


# -------------------------------------------------------------------
# 3) fetch_omdb devuelve algo NO Mapping → se ignora
# -------------------------------------------------------------------
def test_analyze_input_movie_fetch_returns_non_mapping(monkeypatch):
    movie = make_input(title="NonMapping", year=2005)

    def fake_fetch_non_mapping(title, year):
        return ["not", "a", "dict"]

    def fake_extract_ratings(omdb):
        assert omdb == {}  # debe caer a dict vacío
        return 5.0, 10, 50

    def fake_decide_action(**kwargs):
        assert kwargs["imdb_rating"] == 5.0
        assert kwargs["imdb_votes"] == 10
        assert kwargs["rt_score"] == 50
        return "MAYBE", "fallback"

    def fake_detect_misidentified(**kwargs):
        return "check"

    monkeypatch.setattr(analyze_input_core, "extract_ratings_from_omdb", fake_extract_ratings)
    monkeypatch.setattr(analyze_input_core, "decide_action", fake_decide_action)
    monkeypatch.setattr(analyze_input_core, "detect_misidentified", fake_detect_misidentified)

    row = analyze_input_core.analyze_input_movie(movie, fake_fetch_non_mapping)

    assert row["decision"] == "MAYBE"
    assert row["imdb_rating"] == 5.0
    assert row["misidentified_hint"] == "check"


# -------------------------------------------------------------------
# 4) imdb_id_hint presente vs ausente
# -------------------------------------------------------------------
def test_analyze_input_movie_includes_imdb_id_hint(monkeypatch):
    movie = make_input(imdb_id_hint="tt9999")

    monkeypatch.setattr(analyze_input_core, "extract_ratings_from_omdb", lambda _: (None, None, None))
    monkeypatch.setattr(analyze_input_core, "decide_action", lambda **_: ("UNKNOWN", "no ratings"))
    monkeypatch.setattr(analyze_input_core, "detect_misidentified", lambda **_: "")

    row = analyze_input_core.analyze_input_movie(movie, lambda *_: {})

    assert row["imdb_id_hint"] == "tt9999"


def test_analyze_input_movie_omits_imdb_id_hint_when_none(monkeypatch):
    movie = make_input(imdb_id_hint=None)

    monkeypatch.setattr(analyze_input_core, "extract_ratings_from_omdb", lambda _: (None, None, None))
    monkeypatch.setattr(analyze_input_core, "decide_action", lambda **_: ("UNKNOWN", "no ratings"))
    monkeypatch.setattr(analyze_input_core, "detect_misidentified", lambda **_: "")

    row = analyze_input_core.analyze_input_movie(movie, lambda *_: {})

    assert "imdb_id_hint" not in row