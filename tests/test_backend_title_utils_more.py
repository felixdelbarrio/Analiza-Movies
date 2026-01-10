import backend.title_utils as tu


def test_strip_accents_and_cleanup_separators():
    assert tu.strip_accents("\u00c1rbol") == "Arbol"
    assert tu.cleanup_separators("Movie.Title_2020") == "Movie Title 2020"


def test_noise_detection_and_removal(monkeypatch):
    monkeypatch.setattr(tu, "MOVIE_INPUT_LOOKUP_REMOVE_BRACKETED_NOISE", True)
    monkeypatch.setattr(tu, "MOVIE_INPUT_LOOKUP_REMOVE_TRAILING_DASH_GROUP", True)

    assert tu.looks_like_noise_group("1080p") is True
    assert tu.looks_like_noise_group("Movie") is False

    assert tu.remove_bracketed_noise("Movie [1080p]").strip() == "Movie"
    assert tu.remove_trailing_dash_group("Movie - 1080p - x265") == "Movie"

    assert tu.remove_noise_tokens("Movie 1080p x265") == "Movie"


def test_extractors_and_split():
    assert tu.extract_imdb_id_from_text("see tt1234567 here") == "tt1234567"
    assert tu.extract_year_from_text("Movie (1999)") == 1999

    title, year = tu.split_title_and_year_from_text("Movie (2001)")
    assert title == "Movie"
    assert year == 2001

    title2, year2 = tu.split_title_and_year_from_text("Movie 1999 Remux")
    assert title2 == "Movie 1999 Remux"
    assert year2 == 1999


def test_filename_and_clean_title():
    assert tu.filename_stem("/tmp/Movie.1999.mkv") == "Movie.1999"
    assert tu.clean_title_candidate("Movie - 1080p") == "Movie"


def test_normalize_title_for_lookup(monkeypatch):
    monkeypatch.setattr(tu, "MOVIE_INPUT_LOOKUP_STRIP_ACCENTS", True)
    monkeypatch.setattr(tu, "MOVIE_INPUT_LOOKUP_REMOVE_BRACKETED_NOISE", False)
    monkeypatch.setattr(tu, "MOVIE_INPUT_LOOKUP_REMOVE_TRAILING_DASH_GROUP", True)

    out = tu.normalize_title_for_lookup("Am\u00e9lie (2001) - 1080p")
    assert out == "amelie 2001"


def test_normalize_title_for_compare():
    opts = tu.NormalizeOptions(max_len=4, strip_accents=True)
    out = tu.normalize_title_for_compare("\u00c1rbol!!!", options=opts)
    assert out == "arbo"
