import json
from types import SimpleNamespace

from backend import movie_analyzer


def make_movie(title: str = "Sample", year: int = 2020) -> SimpleNamespace:
    return SimpleNamespace(
        librarySectionTitle="MyLib",
        title=title,
        year=year,
        rating=None,
        ratingKey="rk1",
        guid="imdb://tt0000001",
        thumb=None,
        media=[],
    )


def test_analyze_single_movie_minimal(monkeypatch):
    movie = make_movie("My Movie", 2010)

    # Fake get_movie_record that accepts any args/kwargs
    def fake_get_movie_record(*args, **kwargs):
        return {
            "imdbID": "tt0000001",
            "imdbRating": "7.0",
            "imdbVotes": "1,000",
            "Ratings": [{"Source": "Rotten Tomatoes", "Value": "80%"}],
            "Poster": "http://example.com/poster.jpg",
            "Website": "http://example.com",
        }

    monkeypatch.setattr(movie_analyzer, "get_movie_record", fake_get_movie_record)

    # Fake decide_action compatible with any call signature
    def fake_decide_action(*args, **kwargs):
        return "MAYBE", "reason"

    monkeypatch.setattr(movie_analyzer, "decide_action", fake_decide_action)

    # Simple mocks for other helpers
    monkeypatch.setattr(
        movie_analyzer,
        "get_movie_file_info",
        lambda m: ("/tmp/f.mp4", 12345),
    )
    monkeypatch.setattr(
        movie_analyzer,
        "generate_metadata_suggestions_row",
        lambda m, omdb: None,
    )
    monkeypatch.setattr(
        movie_analyzer,
        "detect_misidentified",
        lambda *a, **k: "",
    )

    row, meta, logs = movie_analyzer.analyze_single_movie(movie)

    assert isinstance(row, dict)
    assert row["title"] == "My Movie"
    assert row["imdb_id"] == "tt0000001"
    assert row["decision"] == "MAYBE"
    assert row["file"] == "/tmp/f.mp4"
    assert meta is None
    assert isinstance(logs, list)


def test_analyze_single_movie_with_misidentified_hint(monkeypatch):
    movie = make_movie("Weird Movie", 1999)

    def fake_get_movie_record(*args, **kwargs):
        return {
            "imdbID": "tt0000001",
            "imdbRating": "6.0",
            "imdbVotes": "5,000",
            "Ratings": [],
            "Poster": "http://example.com/poster.jpg",
        }

    monkeypatch.setattr(movie_analyzer, "get_movie_record", fake_get_movie_record)
    monkeypatch.setattr(
        movie_analyzer,
        "decide_action",
        lambda *a, **k: ("MAYBE", "reason"),
    )
    monkeypatch.setattr(
        movie_analyzer,
        "get_movie_file_info",
        lambda m: ("/tmp/weird.mkv", 42),
    )

    # Forzamos una pista de “posible mal identificada”
    monkeypatch.setattr(
        movie_analyzer,
        "detect_misidentified",
        lambda *a, **k: "possible mismatch",
    )

    # Sin sugerencias de metadata en este caso
    monkeypatch.setattr(
        movie_analyzer,
        "generate_metadata_suggestions_row",
        lambda m, omdb: None,
    )

    row, meta, logs = movie_analyzer.analyze_single_movie(movie)

    assert row["title"] == "Weird Movie"
    assert row["misidentified_hint"] == "possible mismatch"
    assert meta is None
    assert any("possible mismatch" in log for log in logs)


def test_analyze_single_movie_with_metadata_suggestion(monkeypatch):
    movie = make_movie("My Movie", 2010)

    def fake_get_movie_record(*args, **kwargs):
        return {
            "imdbID": "tt0000001",
            "imdbRating": "7.0",
            "imdbVotes": "1,000",
            "Ratings": [],
        }

    monkeypatch.setattr(movie_analyzer, "get_movie_record", fake_get_movie_record)
    monkeypatch.setattr(
        movie_analyzer,
        "decide_action",
        lambda *a, **k: ("KEEP", "reason"),
    )
    monkeypatch.setattr(
        movie_analyzer,
        "get_movie_file_info",
        lambda m: ("/tmp/f.mp4", 12345),
    )
    monkeypatch.setattr(
        movie_analyzer,
        "detect_misidentified",
        lambda *a, **k: "",
    )

    meta_row = {"plex_guid": "g1", "action": "Fix title"}

    monkeypatch.setattr(
        movie_analyzer,
        "generate_metadata_suggestions_row",
        lambda m, omdb: meta_row,
    )

    row, meta, logs = movie_analyzer.analyze_single_movie(movie)

    assert row["title"] == "My Movie"
    assert row["imdb_id"] == "tt0000001"
    assert row["decision"] == "KEEP"
    assert meta is meta_row
    assert isinstance(logs, list)


def test_analyze_single_movie_without_file_info(monkeypatch):
    movie = make_movie("NoFile", 2000)

    def fake_get_movie_record(*args, **kwargs):
        return {
            "imdbID": "tt0000002",
            "imdbRating": "6.5",
            "imdbVotes": "500",
            "Ratings": [],
        }

    monkeypatch.setattr(movie_analyzer, "get_movie_record", fake_get_movie_record)
    monkeypatch.setattr(
        movie_analyzer,
        "decide_action",
        lambda *a, **k: ("MAYBE", "reason"),
    )

    # Sin info de fichero
    monkeypatch.setattr(
        movie_analyzer,
        "get_movie_file_info",
        lambda m: (None, None),
    )
    monkeypatch.setattr(
        movie_analyzer,
        "generate_metadata_suggestions_row",
        lambda m, omdb: None,
    )
    monkeypatch.setattr(
        movie_analyzer,
        "detect_misidentified",
        lambda *a, **k: "",
    )

    row, meta, logs = movie_analyzer.analyze_single_movie(movie)

    assert row["title"] == "NoFile"
    # El código actual normaliza path faltante a cadena vacía
    assert row["file"] == ""
    assert row["file_size"] is None
    assert row["imdb_id"] == "tt0000002"
    assert meta is None
    assert isinstance(logs, list)


def test_analyze_single_movie_omdb_json_is_valid(monkeypatch):
    movie = make_movie("Json Movie", 2015)

    def fake_get_movie_record(*args, **kwargs):
        return {
            "imdbID": "tt1234567",
            "imdbRating": "8.0",
            "imdbVotes": "10,000",
            "Ratings": [],
            "ExtraField": "whatever",
        }

    monkeypatch.setattr(movie_analyzer, "get_movie_record", fake_get_movie_record)
    monkeypatch.setattr(
        movie_analyzer,
        "decide_action",
        lambda *a, **k: ("KEEP", "reason"),
    )
    monkeypatch.setattr(
        movie_analyzer,
        "get_movie_file_info",
        lambda m: ("/tmp/json.mkv", 42),
    )
    monkeypatch.setattr(
        movie_analyzer,
        "generate_metadata_suggestions_row",
        lambda m, omdb: None,
    )
    monkeypatch.setattr(
        movie_analyzer,
        "detect_misidentified",
        lambda *a, **k: "",
    )

    row, meta, logs = movie_analyzer.analyze_single_movie(movie)

    assert "omdb_json" in row
    parsed = json.loads(row["omdb_json"])
    assert parsed["imdbID"] == "tt1234567"
    assert meta is None
    assert isinstance(logs, list)