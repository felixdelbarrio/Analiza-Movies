import pandas as pd

from frontend.data_utils import (
    add_derived_columns,
    build_word_counts,
    explode_genres_from_omdb_json,
    format_count_size,
    safe_json_loads_single,
)


def test_safe_json_loads_single():
    assert safe_json_loads_single({"a": 1}) == {"a": 1}
    assert safe_json_loads_single("{\"a\": 1}") == {"a": 1}
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


def test_explode_genres_from_omdb_json():
    df = pd.DataFrame(
        [
            {"title": "Movie", "omdb_json": '{"Genre": "Action, Drama"}'},
            {"title": "NoGenres", "omdb_json": "{}"},
        ]
    )
    exploded = explode_genres_from_omdb_json(df)

    assert set(exploded["genre"].tolist()) == {"Action", "Drama"}


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
