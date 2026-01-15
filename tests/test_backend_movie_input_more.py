import backend.movie_input as mi
import backend.title_utils as tu


def test_coalesce_movie_identity_from_path():
    title, year, imdb_id = mi.coalesce_movie_identity(
        title="",
        year=None,
        file_path="/movies/Best.Movie.1999.tt1234567.mkv",
        imdb_id_hint=None,
    )

    assert title.startswith("Best Movie")
    assert year == 1999
    assert imdb_id == "tt1234567"


def test_guess_language_from_title_or_path(monkeypatch):
    monkeypatch.setattr(mi, "MOVIE_INPUT_LANG_FUNCTION_WORD_MIN_HITS", 1)
    assert mi.guess_spanish_from_title_or_path("El Viaje", "") is True
    assert mi.guess_italian_from_title_or_path("", "/path/film.italiano.mkv") is True
    assert mi.guess_french_from_title_or_path("", "/path/vostfr/movie.mkv") is True
    assert mi.guess_japanese_from_title_or_path("\u65e5\u672c\u8a9e", "") is True
    assert mi.guess_korean_from_title_or_path("\ud55c\uad6d\uc5b4", "") is True
    assert mi.guess_chinese_from_title_or_path("\u4e2d\u6587", "") is True

    monkeypatch.setattr(mi, "MOVIE_INPUT_LANG_SKIP_ENGLISH_IF_CJK", True)
    assert (
        mi.guess_english_from_title_or_path("The Movie \u65e5\u672c\u8a9e", "") is False
    )


def test_detect_context_language_code_prefers_library_language():
    movie = mi.MovieInput(
        source="plex",
        library="Test",
        title="Movie",
        year=2020,
        file_path="",
        file_size_bytes=None,
        imdb_id_hint=None,
        plex_guid=None,
        rating_key=None,
        thumb_url=None,
        extra={"library_language": "es"},
    )
    assert mi.detect_context_language_code(movie) == "es"


def test_should_skip_new_title_suggestion(monkeypatch):
    monkeypatch.setattr(tu, "MOVIE_INPUT_LANG_FUNCTION_WORD_MIN_HITS", 1)
    assert (
        mi.should_skip_new_title_suggestion(
            context_lang="es",
            current_title="La Pelicula",
            omdb_title="The Movie",
        )
        is True
    )

    assert (
        mi.should_skip_new_title_suggestion(
            context_lang="en",
            current_title="The Movie",
            omdb_title="The Movie",
        )
        is False
    )


def test_movie_input_helpers():
    movie = mi.MovieInput(
        source="dlna",
        library="Library",
        title="Movie",
        year=2001,
        file_path="/movies/Movie.mkv",
        file_size_bytes=123,
        imdb_id_hint=None,
        plex_guid=None,
        rating_key=None,
        thumb_url=None,
        extra={},
    )

    assert movie.has_physical_file() is True
    assert movie.normalized_title() == "movie"
    assert movie.normalized_title_for_lookup() == "movie"
    assert "Movie (2001)" in movie.describe()
