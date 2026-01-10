import backend.omdb_client as oc


def test_safe_cast_helpers():
    assert oc._int_or_none("1,200") == 1200
    assert oc._int_or_none("N/A") is None
    assert oc._float_or_none("7.5") == 7.5
    assert oc._float_or_none("bad") is None


def test_cache_keys_and_expiry():
    assert oc._ty_key("title", "1999") == "title|1999"
    assert oc._cache_key_for_imdb("TT123") == "imdb:tt123"
    assert oc._cache_key_for_title_year("t", "y") == "ty:t|y"
    assert oc._cache_key_for_title_only("t") == "t:t"
    assert oc._rid_for_record(imdb_norm="tt1", norm_title="t", norm_year="y") == "imdb:tt1"
    assert oc._rid_for_record(imdb_norm=None, norm_title="t", norm_year="y") == "ty:t|y"

    item = {"fetched_at": 100, "ttl_s": 10, "Title": "x", "Year": "1999", "imdbID": None, "omdb": {}, "status": "ok"}
    assert oc._is_expired_item(item, 200) is True
    assert oc._is_expired_item(item, 105) is False
