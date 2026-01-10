import backend.wiki_client as wc


def test_basic_helpers():
    assert wc._normalize_lang_code("SPA") == "es"
    assert wc._normalize_lang_code("en-US") == "en"
    assert wc._norm_imdb(" TT123 ") == "tt123"
    assert wc._norm_imdb(None) is None

    assert wc._ty_key("title", "1999") == "title|1999"
    assert wc._request_key_for_singleflight(imdb_norm="tt1", norm_title="t", norm_year="y") == "imdb:tt1"
    assert wc._request_key_for_singleflight(imdb_norm=None, norm_title="t", norm_year="y") == "ty:t|y"

    assert wc._is_expired(100, 10, 200) is True
    assert wc._is_expired(100, 200, 200) is False
