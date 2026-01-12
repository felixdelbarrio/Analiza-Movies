import backend.scoring as scoring


def test_compute_scoring_no_data_and_meta_only():
    out = scoring.compute_scoring(None, None, None)
    assert out["decision"] == "UNKNOWN"
    assert out["rule"] == "NO_DATA"

    out2 = scoring.compute_scoring(None, None, None, metacritic_score=80)
    assert out2["decision"] == "UNKNOWN"
    assert out2["rule"] == "META_ONLY"


def test_compute_scoring_rt_boost(monkeypatch):
    monkeypatch.setattr(scoring, "get_auto_keep_rating_threshold", lambda: 7.0)
    monkeypatch.setattr(scoring, "get_auto_delete_rating_threshold", lambda: 5.5)
    monkeypatch.setattr(scoring, "get_votes_threshold_for_year", lambda year: 100)
    monkeypatch.setattr(scoring, "get_global_imdb_mean_from_cache", lambda: 6.5)

    out = scoring.compute_scoring(7.2, 1000, 90, year=2010)
    assert out["decision"] == "KEEP"
    assert out["rule"] == "KEEP_RT_BOOST"


def test_compute_scoring_rt_tiebreaker(monkeypatch):
    monkeypatch.setattr(scoring, "get_auto_keep_rating_threshold", lambda: 7.0)
    monkeypatch.setattr(scoring, "get_auto_delete_rating_threshold", lambda: 5.5)
    monkeypatch.setattr(scoring, "get_votes_threshold_for_year", lambda year: 100)
    monkeypatch.setattr(scoring, "get_global_imdb_mean_from_cache", lambda: 6.5)

    out = scoring.compute_scoring(5.0, 100, 10, year=2000)
    assert out["decision"] == "DELETE"
    assert out["rule"] == "DELETE_RT_TIEBREAKER"


def test_compute_scoring_maybe_low_info():
    out = scoring.compute_scoring(6.0, None, None)
    assert out["decision"] == "MAYBE"
    assert out["rule"] == "MAYBE_LOW_INFO"
