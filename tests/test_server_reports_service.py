import pandas as pd

from server.api.services import reports
from server.api.services.reports import (
    df_to_page,
    enrich_report_context,
    prepare_search_blob,
)


def test_prepare_search_blob_and_df_to_page_query():
    df = pd.DataFrame(
        [
            {
                "title": "La guerra de los mundos",
                "original_title": "War of the Worlds",
                "title_aliases": '["War of the Worlds"]',
                "imdb_id": "tt123",
                "file": "/movies/War.of.the.Worlds.2005.mkv",
            },
            {"title": "Other", "imdb_id": "tt999", "file": "B"},
        ]
    )
    prepare_search_blob(df)
    assert "__search_blob" in df.columns

    page = df_to_page(df, offset=0, limit=10, query="war of the worlds")
    assert page["total"] == 1
    assert page["items"][0]["title"] == "La guerra de los mundos"


def test_enrich_report_context_merges_cached_editorial_fields(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "title": "Movie",
                "year": "1999",
                "imdb_id": "tt123",
                "director": "",
                "actors": "",
                "genre": "",
                "plot": "",
                "wikipedia_title": "",
                "wikipedia_summary": "",
            }
        ]
    )

    monkeypatch.setattr(
        reports,
        "load_omdb_payload",
        lambda _cache, _profile_id=None: {
            "records": {
                "omdb-1": {
                    "imdbID": "tt123",
                    "omdb": {
                        "Director": "Jane Doe",
                        "Actors": "Actor One, Actor Two",
                        "Genre": "Drama",
                        "Plot": "A detailed plot.",
                        "Writer": "Writer One",
                        "Runtime": "101 min",
                    },
                }
            },
            "index_imdb": {"tt123": "omdb-1"},
        },
    )
    monkeypatch.setattr(
        reports,
        "load_wiki_payload",
        lambda _cache, _profile_id=None: {
            "records": {
                "wiki-1": {
                    "imdbID": "tt123",
                    "wiki": {
                        "wikipedia_title": "Movie Wiki",
                        "source_language": "es",
                        "summary": "Wiki summary",
                    },
                    "wikidata": {"qid": "Q1"},
                }
            },
            "index_imdb": {"tt123": "wiki-1"},
        },
    )

    enrich_report_context(df, cache=object(), profile_id="profile-1")
    prepare_search_blob(df)

    assert df.loc[0, "director"] == "Jane Doe"
    assert df.loc[0, "actors"] == "Actor One, Actor Two"
    assert df.loc[0, "genre"] == "Drama"
    assert df.loc[0, "plot"] == "A detailed plot."
    assert df.loc[0, "wikipedia_title"] == "Movie Wiki"
    assert df.loc[0, "wikipedia_summary"] == "Wiki summary"
    assert df.loc[0, "wikidata_id"] == "Q1"
    assert df.loc[0, "source_language"] == "es"
    assert "Actor One" in str(df.loc[0, "search_context"])

    page = df_to_page(df, offset=0, limit=10, query="actor two")
    assert page["total"] == 1

    summary_page = df_to_page(df, offset=0, limit=10, query="wiki summary")
    assert summary_page["total"] == 1


def test_enrich_report_context_fills_nan_text_columns_without_dtype_errors(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "title": "Movie",
                "year": "1999",
                "imdb_id": "tt123",
                "director": float("nan"),
                "actors": float("nan"),
                "genre": float("nan"),
                "plot": float("nan"),
                "wikipedia_title": float("nan"),
                "wikipedia_summary": float("nan"),
                "source_language": float("nan"),
            }
        ]
    )

    monkeypatch.setattr(
        reports,
        "load_omdb_payload",
        lambda _cache, _profile_id=None: {
            "records": {
                "omdb-1": {
                    "imdbID": "tt123",
                    "omdb": {
                        "Director": "Jane Doe",
                        "Actors": "Actor One, Actor Two",
                        "Genre": "Drama",
                        "Plot": "A detailed plot.",
                    },
                }
            },
            "index_imdb": {"tt123": "omdb-1"},
        },
    )
    monkeypatch.setattr(
        reports,
        "load_wiki_payload",
        lambda _cache, _profile_id=None: {
            "records": {
                "wiki-1": {
                    "imdbID": "tt123",
                    "wiki": {
                        "wikipedia_title": "Movie Wiki",
                        "source_language": "es",
                        "summary": "Wiki summary",
                    },
                    "wikidata": {"qid": "Q1"},
                }
            },
            "index_imdb": {"tt123": "wiki-1"},
        },
    )

    enrich_report_context(df, cache=object(), profile_id="profile-1")

    assert str(df["director"].dtype) == "string"
    assert df.loc[0, "director"] == "Jane Doe"
    assert df.loc[0, "actors"] == "Actor One, Actor Two"
    assert df.loc[0, "genre"] == "Drama"
    assert df.loc[0, "plot"] == "A detailed plot."
    assert df.loc[0, "wikipedia_title"] == "Movie Wiki"
    assert df.loc[0, "wikipedia_summary"] == "Wiki summary"
    assert df.loc[0, "wikidata_id"] == "Q1"
    assert df.loc[0, "source_language"] == "es"
