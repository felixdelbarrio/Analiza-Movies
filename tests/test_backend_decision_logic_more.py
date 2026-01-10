import backend.decision_logic as dl


def test_extract_omdb_year():
    assert dl._extract_omdb_year("1999-2001") == 1999
    assert dl._extract_omdb_year("N/A") is None


def test_is_omdb_usable_when_response_true():
    assert dl._is_omdb_usable({"Response": "True"}) is True


def test_is_omdb_usable_without_response(monkeypatch):
    monkeypatch.setattr(dl, "_OMDB_REQUIRE_RESPONSE_TRUE", False)
    assert dl._is_omdb_usable({"imdbID": "tt123"}) is True
    assert dl._is_omdb_usable({"Title": "Movie"}) is True
    assert dl._is_omdb_usable({}) is False


def test_evaluate_rules_imdb_match_metadata_diverge():
    omdb = {"Response": "True", "imdbID": "tt1234567", "Title": "Movie", "Year": "2005"}
    hits = dl.evaluate_misidentified_rules(
        plex_title="Movie",
        plex_year=2000,
        plex_imdb_id="tt1234567",
        omdb_data=omdb,
        imdb_rating=7.0,
        imdb_votes=1000,
        rt_score=None,
    )
    assert hits
    assert hits[0].rule_id == dl.RULE_IMDB_MATCH_METADATA_DIVERGE


def test_evaluate_rules_imdb_mismatch_and_title_mismatch():
    omdb = {"Response": "True", "imdbID": "tt999", "Title": "WXYZ", "Year": "1999"}
    hits = dl.evaluate_misidentified_rules(
        plex_title="ABCD",
        plex_year=1999,
        plex_imdb_id="tt123",
        omdb_data=omdb,
        imdb_rating=None,
        imdb_votes=None,
        rt_score=None,
    )
    assert any(hit.rule_id == dl.RULE_IMDB_MISMATCH for hit in hits)

    omdb2 = {"Response": "True", "Title": "WXYZ", "Year": "1999"}
    hits2 = dl.evaluate_misidentified_rules(
        plex_title="ABCD",
        plex_year=1999,
        plex_imdb_id=None,
        omdb_data=omdb2,
        imdb_rating=None,
        imdb_votes=None,
        rt_score=None,
    )
    assert any(hit.rule_id == dl.RULE_TITLE_MISMATCH for hit in hits2)


def test_summarize_rule_hits_sorted():
    hits = [
        dl.RuleHit(dl.RULE_LOW_RT_KNOWN, "rt", "soft", "soft"),
        dl.RuleHit(dl.RULE_IMDB_MISMATCH, "imdb", "hard", "hard"),
    ]
    out = dl.summarize_rule_hits(hits)
    assert out.startswith("hard")


def test_sort_filtered_rows_orders_decisions_and_votes():
    rows = [
        {"decision": "KEEP", "imdb_votes": 10, "title": "b"},
        {"decision": "DELETE", "imdb_votes": 1, "title": "a"},
        {"decision": "DELETE", "imdb_votes": 100, "title": "c"},
    ]
    ordered = dl.sort_filtered_rows(rows)
    assert ordered[0]["title"] == "c"
    assert ordered[1]["title"] == "a"
    assert ordered[2]["title"] == "b"
