import backend.analyze_input_core as core
from backend.movie_input import MovieInput


def test_normalize_decision_and_reason():
    assert core._normalize_decision("keep") == "KEEP"
    assert core._normalize_decision("invalid") == "UNKNOWN"

    assert core._normalize_reason("") == "scoring did not provide a usable reason"
    assert core._normalize_reason(None) == "scoring did not provide a usable reason"


def test_coerce_ratings_tuple():
    out = core._coerce_ratings_tuple((7.1, 1000, 80, 75))
    assert out == (7.1, 1000, 80, 75)

    out2 = core._coerce_ratings_tuple((7.1, 1000))
    assert out2 == (7.1, 1000, None, None)

    out3 = core._coerce_ratings_tuple("bad")
    assert out3 == (None, None, None, None)


def test_derive_reason_code():
    rc = core._derive_reason_code(
        scoring=None,
        decision="KEEP",
        used_omdb=False,
        omdb_usable=False,
        has_signals=False,
    )
    assert rc == "strong_without_external"

    rc2 = core._derive_reason_code(
        scoring=None,
        decision="UNKNOWN",
        used_omdb=True,
        omdb_usable=True,
        has_signals=False,
    )
    assert rc2 == "external_unusable_no_signals"


def test_get_lookup_identity_fallback(monkeypatch):
    movie = MovieInput(
        source="dlna",
        library="Lib",
        title="",
        year=None,
        file_path="/movies/Best.Movie.1999.mkv",
        file_size_bytes=None,
        imdb_id_hint=None,
        plex_guid=None,
        rating_key=None,
        thumb_url=None,
        extra={"source_url": "http://x/tt1234567"},
    )

    traces = []

    def _trace(msg: str) -> None:
        traces.append(msg)

    monkeypatch.setattr(core, "_LOOKUP_TITLE_FALLBACK_ENABLED", True)
    title, year, imdb_id = core._get_lookup_identity(movie, trace=_trace)

    assert title == "tt1234567"
    assert year == 1999
    assert imdb_id == "tt1234567"
    assert any("fallback" in msg for msg in traces)
