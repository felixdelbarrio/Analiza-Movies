from backend.scoring import compute_scoring


def test_compute_scoring_no_data_returns_unknown():
    result = compute_scoring(None, None, None)
    assert result["decision"] == "UNKNOWN"
    assert result["rule"] == "NO_DATA"


def test_compute_scoring_metacritic_only_returns_unknown():
    result = compute_scoring(None, None, None, metacritic_score=70)
    assert result["decision"] == "UNKNOWN"
    assert result["rule"] == "META_ONLY"
