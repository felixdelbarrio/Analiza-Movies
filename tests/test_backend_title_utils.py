from backend.title_utils import (
    clean_title_candidate,
    extract_imdb_id_from_text,
    extract_year_from_text,
    filename_stem,
    split_title_and_year_from_text,
)


def test_extract_imdb_id_from_text_case_insensitive():
    assert extract_imdb_id_from_text("https://imdb.com/title/TT1234567/") == "tt1234567"
    assert extract_imdb_id_from_text("no id here") is None


def test_extract_year_and_split_title():
    assert extract_year_from_text("Movie 1999 Remux") == 1999
    assert extract_year_from_text("No year") is None

    title, year = split_title_and_year_from_text("Best Movie (2001)")
    assert title == "Best Movie"
    assert year == 2001

    title, year = split_title_and_year_from_text("Another.Movie.2005")
    assert title == "Another.Movie"
    assert year == 2005


def test_filename_stem_and_clean_title_candidate():
    assert filename_stem("/movies/Best.Movie.1999.mkv") == "Best.Movie.1999"
    cleaned = clean_title_candidate("Best.Movie.1999 [1080p]")
    assert cleaned.startswith("Best Movie")
