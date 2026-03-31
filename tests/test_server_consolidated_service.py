from server.api.services import consolidated


class DummyCache:
    pass


def test_consolidate_merges_omdb_and_wiki(monkeypatch):
    def _omdb_payload(_cache, profile_id=None):
        return {
            "records": {
                "1": {
                    "omdb": {
                        "Title": "Movie",
                        "Year": "1999",
                        "imdbRating": "7.0",
                        "__wiki": {"wikipedia_title": "Movie Wiki"},
                    },
                    "imdbID": "tt123",
                }
            },
            "index_imdb": {"tt123": "1"},
        }

    def _wiki_payload(_cache, profile_id=None):
        return {
            "records": {
                "2": {
                    "wiki": {
                        "wikipedia_title": "Movie Wiki",
                        "source_language": "en",
                        "summary": "Localized summary",
                    },
                    "wikidata": {"wikidata_id": "Q1"},
                }
            },
            "index_imdb": {"tt123": "2"},
        }

    monkeypatch.setattr(consolidated, "load_omdb_payload", _omdb_payload)
    monkeypatch.setattr(consolidated, "load_wiki_payload", _wiki_payload)

    out = consolidated.consolidate(
        cache=DummyCache(), imdb_id="tt123", title="Movie", year="1999"
    )

    assert out["merged"]["title"] == "Movie"
    assert out["merged"]["wikipedia_title"] == "Movie Wiki"
    assert out["merged"]["wikipedia_summary"] == "Localized summary"
    assert out["merged"]["wikidata_id"] == "Q1"


def test_consolidate_normalizes_title_for_lookup(monkeypatch):
    def _omdb_payload(_cache, profile_id=None):
        return {
            "records": {
                "1": {
                    "omdb": {"Title": "Machine of War", "Year": "2026"},
                    "imdbID": "tt999",
                }
            },
            "index_imdb": {},
            "index_ty": {"maquina de guerra|2026": "1"},
        }

    def _wiki_payload(_cache, profile_id=None):
        return {"records": {}, "index_imdb": {}, "index_ty": {}}

    monkeypatch.setattr(consolidated, "load_omdb_payload", _omdb_payload)
    monkeypatch.setattr(consolidated, "load_wiki_payload", _wiki_payload)

    out = consolidated.consolidate(
        cache=DummyCache(), imdb_id=None, title="Máquina de guerra", year="2026"
    )

    assert out["sources"]["omdb"]["rid"] == "1"
    assert out["key"]["title_norm"] == "maquina de guerra"
