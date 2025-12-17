# tests/test_dlna_input.py
from __future__ import annotations

from backend.movie_input import MovieInput


# ----------------------------------------------------------------------
# Construcción básica del dataclass
# ----------------------------------------------------------------------


def test_dlna_input_creation_basic() -> None:
    obj = MovieInput(
        source="dlna",
        library="My Videos",
        title="The Movie",
        year=2020,
        file_path="/tmp/file.mkv",
        file_size_bytes=123456,
        imdb_id_hint="tt1234567",
        plex_guid=None,
        rating_key=None,
        thumb_url=None,
    )

    assert obj.source == "dlna"
    assert obj.library == "My Videos"
    assert obj.title == "The Movie"
    assert obj.year == 2020
    assert obj.file_path == "/tmp/file.mkv"
    assert obj.file_size_bytes == 123456
    assert obj.imdb_id_hint == "tt1234567"
    assert obj.plex_guid is None
    assert obj.rating_key is None
    assert obj.thumb_url is None
    assert obj.extra == {}  # default_factory


# ----------------------------------------------------------------------
# has_physical_file()
# ----------------------------------------------------------------------


def test_dlna_has_physical_file_true() -> None:
    obj = MovieInput(
        source="local",
        library="Docs",
        title="Example",
        year=2020,
        file_path="/tmp/movie.mp4",
        file_size_bytes=None,
        imdb_id_hint=None,
        plex_guid=None,
        rating_key=None,
        thumb_url=None,
    )
    assert obj.has_physical_file() is True


def test_dlna_has_physical_file_false() -> None:
    obj = MovieInput(
        source="local",
        library="Docs",
        title="Example",
        year=2020,
        file_path="",  # vacío → no fichero
        file_size_bytes=None,
        imdb_id_hint=None,
        plex_guid=None,
        rating_key=None,
        thumb_url=None,
    )
    assert obj.has_physical_file() is False


# ----------------------------------------------------------------------
# normalized_title()
# ----------------------------------------------------------------------


def test_dlna_normalized_title_basic() -> None:
    obj = MovieInput(
        source="plex",
        library="Movies",
        title="  La PELI   ",
        year=2010,
        file_path="/tmp/a.mp4",
        file_size_bytes=None,
        imdb_id_hint=None,
        plex_guid=None,
        rating_key=None,
        thumb_url=None,
    )
    assert obj.normalized_title() == "la peli"


def test_dlna_normalized_title_empty() -> None:
    obj = MovieInput(
        source="plex",
        library="Movies",
        title="   ",
        year=None,
        file_path="path",
        file_size_bytes=None,
        imdb_id_hint=None,
        plex_guid=None,
        rating_key=None,
        thumb_url=None,
    )
    # "   ".lower().strip() → ""
    assert obj.normalized_title() == ""


# ----------------------------------------------------------------------
# describe()
# ----------------------------------------------------------------------


def test_dlna_describe_with_file() -> None:
    obj = MovieInput(
        source="plex",
        library="Movies",
        title="Matrix",
        year=1999,
        file_path="/movies/matrix.mkv",
        file_size_bytes=None,
        imdb_id_hint=None,
        plex_guid="plex://123",
        rating_key="rk1",
        thumb_url=None,
    )
    desc = obj.describe()
    assert "[plex]" in desc
    assert "Matrix (1999)" in desc
    assert "/movies/matrix.mkv" in desc
    assert "Movies" in desc


def test_dlna_describe_without_file() -> None:
    obj = MovieInput(
        source="dlna",
        library="DLNA Lib",
        title="Something",
        year=None,
        file_path="",
        file_size_bytes=None,
        imdb_id_hint=None,
        plex_guid=None,
        rating_key=None,
        thumb_url=None,
    )
    desc = obj.describe()

    assert "[dlna]" in desc
    assert "Something (?)" in desc
    assert "/ " in desc or "/?"  # descripción mínima
    assert obj.file_path == ""  # sin file


# ----------------------------------------------------------------------
# extra field integrity
# ----------------------------------------------------------------------


def test_dlna_extra_dict_is_independent() -> None:
    obj1 = MovieInput(
        source="local",
        library="Lib",
        title="A",
        year=2000,
        file_path="f",
        file_size_bytes=None,
        imdb_id_hint=None,
        plex_guid=None,
        rating_key=None,
        thumb_url=None,
    )

    obj2 = MovieInput(
        source="local",
        library="Lib",
        title="B",
        year=2001,
        file_path="g",
        file_size_bytes=None,
        imdb_id_hint=None,
        plex_guid=None,
        rating_key=None,
        thumb_url=None,
    )

    obj1.extra["k"] = 1
    assert obj2.extra == {}  # deben ser independientes


def test_dlna_create_with_extra_initial_data() -> None:
    obj = MovieInput(
        source="other",
        library="X",
        title="Z",
        year=2022,
        file_path="f",
        file_size_bytes=None,
        imdb_id_hint=None,
        plex_guid=None,
        rating_key=None,
        thumb_url=None,
        extra={"foo": 42},
    )
    assert obj.extra == {"foo": 42}