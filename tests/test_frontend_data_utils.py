import pandas as pd

import frontend.data_utils as data_utils
from frontend.data_utils import (
    add_derived_columns,
    build_word_counts,
    explode_genres_from_omdb_json,
    format_count_size,
    directors_from_omdb_json_or_cache,
    safe_json_loads_single,
)


def test_safe_json_loads_single():
    assert safe_json_loads_single({"a": 1}) == {"a": 1}
    assert safe_json_loads_single('{"a": 1}') == {"a": 1}
    assert safe_json_loads_single("not-json") is None


def test_add_derived_columns_metacritic_and_sizes():
    df = pd.DataFrame(
        [
            {
                "omdb_json": '{"Metascore": "68", "Ratings": []}',
                "file_size": 1024**3,
                "year": 1999,
            }
        ]
    )
    out = add_derived_columns(df)

    assert out.loc[0, "metacritic_score"] == 68.0
    assert out.loc[0, "file_size_gb"] == 1.0
    assert out.loc[0, "decade_label"] == "1990s"


def test_add_derived_columns_numeric_coercion():
    df = pd.DataFrame(
        [
            {
                "imdb_rating": "7.2",
                "rt_score": "91",
                "metacritic_score": "84",
                "year": "2004",
            }
        ]
    )
    out = add_derived_columns(df)

    assert out.loc[0, "imdb_rating"] == 7.2
    assert out.loc[0, "rt_score"] == 91.0
    assert out.loc[0, "metacritic_score"] == 84.0
    assert out.loc[0, "decade_label"] == "2000s"


def test_explode_genres_from_omdb_json():
    df = pd.DataFrame(
        [
            {"title": "Movie", "omdb_json": '{"Genre": "Action, Drama"}'},
            {"title": "NoGenres", "omdb_json": "{}"},
        ]
    )
    exploded = explode_genres_from_omdb_json(df)

    assert set(exploded["genre"].tolist()) == {"Action", "Drama"}


def test_explode_genres_from_omdb_cache_fallback(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "title": "Movie",
                "imdb_id": "tt123",
                "omdb_json": "",
            }
        ]
    )
    monkeypatch.setattr(
        data_utils,
        "_load_omdb_cache_genres",
        lambda: {"tt123": ["Action", "Drama"]},
    )

    exploded = explode_genres_from_omdb_json(df)

    assert set(exploded["genre"].tolist()) == {"Action", "Drama"}


def test_directors_from_omdb_json_or_cache_fallback(monkeypatch):
    monkeypatch.setattr(
        data_utils,
        "_load_omdb_cache_directors",
        lambda: {"tt123": ["Director A", "Director B"]},
    )

    directors = directors_from_omdb_json_or_cache("", "tt123")

    assert directors == ["Director A", "Director B"]


def test_directors_from_omdb_json_or_cache_from_json():
    directors = directors_from_omdb_json_or_cache({"Director": "A, B"}, "tt123")

    assert directors == ["A", "B"]


def test_split_values_helpers():
    assert data_utils._split_genre_value("Action, Drama") == ["Action", "Drama"]
    assert data_utils._split_genre_value("N/A") == []
    assert data_utils._split_genre_value("") == []

    assert data_utils._split_director_value("A, B") == ["A", "B"]
    assert data_utils._split_director_value("N/A") == []
    assert data_utils._split_director_value("") == []


def test_parse_metacritic_value_variants():
    assert data_utils._parse_metacritic_value("68/100") == 68.0
    assert data_utils._parse_metacritic_value("68") == 68.0
    assert data_utils._parse_metacritic_value(68) == 68.0
    assert data_utils._parse_metacritic_value("N/A") is None
    assert data_utils._parse_metacritic_value("bad") is None


def test_build_word_counts_and_format_count_size():
    df = pd.DataFrame(
        [
            {"title": "The Big Movie", "decision": "DELETE"},
            {"title": "Big Adventure", "decision": "DELETE"},
            {"title": "Small Story", "decision": "KEEP"},
        ]
    )
    counts = build_word_counts(df, decisions=["DELETE"])

    assert "big" in counts["word"].tolist()
    assert counts["decision"].unique().tolist() == ["DELETE"]

    assert format_count_size(2, 1.5) == "2 (1.50 GB)"
    assert format_count_size(0, None) == "0"
