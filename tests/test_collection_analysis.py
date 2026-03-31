from __future__ import annotations

from backend import collection_analysis
from backend.movie_input import MovieInput


def test_analyze_movie_uses_prefetched_plex_metadata_without_source_movie(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_analyze_input_movie(movie_input, fetch_omdb, **kwargs):
        del movie_input, fetch_omdb
        captured["plex_rating"] = kwargs.get("plex_rating")
        captured["plex_title"] = kwargs.get("plex_title")
        return {"decision": "KEEP", "file_size_bytes": 123}

    monkeypatch.setattr(
        collection_analysis,
        "analyze_input_movie",
        _fake_analyze_input_movie,
    )
    monkeypatch.setattr(
        collection_analysis,
        "_should_fetch_wiki_for_reporting",
        lambda base_row: False,
    )
    monkeypatch.setattr(
        collection_analysis,
        "generate_metadata_suggestions_row",
        lambda movie_input, omdb_dict=None: {"library": movie_input.library},
    )

    movie_input = MovieInput(
        source="plex",
        library="Sagas",
        title="Star Wars V",
        year=1980,
        file_path="/movies/star-wars-v.mkv",
        file_size_bytes=123,
        imdb_id_hint=None,
        plex_guid="plex://movie/test",
        rating_key="42",
        thumb_url="/thumb.jpg",
        extra={
            "display_title": "El imperio contraataca",
            "display_year": 1980,
            "plex_original_title": "Star Wars Episode V - The Empire Strikes Back",
            "plex_user_rating": 8.5,
            "plex_rating": 7.2,
            "library_language": "es",
        },
    )

    row, meta_sugg, _logs = collection_analysis.analyze_movie(movie_input)

    assert captured["plex_rating"] == 8.5
    assert captured["plex_title"] == "Star Wars Episode V - The Empire Strikes Back"
    assert row is not None
    assert row["title"] == "El imperio contraataca"
    assert meta_sugg == {"library": "Sagas"}
