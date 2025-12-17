from types import SimpleNamespace

from backend import plex_client


def make_part(file_path=None, size=None):
    return SimpleNamespace(file=file_path, size=size)


def make_media(parts):
    return SimpleNamespace(parts=parts)


# -------------------------------------------------------------
# get_movie_file_info
# -------------------------------------------------------------


def test_get_movie_file_info_basic():
    # media with two parts
    p1 = make_part("/tmp/movie1.mkv", 1024)
    p2 = make_part("/tmp/movie1_cd2.mkv", 2048)
    movie = SimpleNamespace(media=[make_media([p1, p2])])
    path, size = plex_client.get_movie_file_info(movie)
    assert path == "/tmp/movie1.mkv"
    assert size == 3072


def test_get_movie_file_info_missing_parts():
    movie = SimpleNamespace(media=None)
    path, size = plex_client.get_movie_file_info(movie)
    assert path is None and size is None


def test_get_movie_file_info_multiple_media_and_invalid_parts():
    # Primer media con una parte inválida y una válida, segundo media con partes válidas
    invalid_part = make_part(file_path=None, size=1000)         # sin ruta → se ignora
    valid_part_1 = make_part("/tmp/m1.mkv", 1000)
    valid_part_2 = make_part("/tmp/m1_cd2.mkv", 2000)
    valid_part_3 = make_part("/tmp/m1_extra.mkv", 3000)

    media1 = make_media([invalid_part, valid_part_1])
    media2 = make_media([valid_part_2, valid_part_3])

    movie = SimpleNamespace(media=[media1, media2])

    path, size = plex_client.get_movie_file_info(movie)

    # Debe quedarse con la primera ruta válida encontrada
    assert path == "/tmp/m1.mkv"
    # Y sumar todos los tamaños válidos
    assert size == 1000 + 2000 + 3000


def test_get_movie_file_info_all_invalid_parts():
    # Todas las parts son inválidas (file no str o size no int)
    p1 = make_part(file_path=None, size=1024)
    p2 = make_part(file_path=123, size="2048")
    movie = SimpleNamespace(media=[make_media([p1, p2])])

    path, size = plex_client.get_movie_file_info(movie)
    assert path is None
    assert size is None


# -------------------------------------------------------------
# get_imdb_id_from_plex_guid
# -------------------------------------------------------------


def test_get_imdb_id_from_plex_guid_variants():
    assert (
        plex_client.get_imdb_id_from_plex_guid(
            "com.plexapp.agents.imdb://tt1234567?lang=en"
        )
        == "tt1234567"
    )
    assert plex_client.get_imdb_id_from_plex_guid("imdb://tt7654321") == "tt7654321"
    assert plex_client.get_imdb_id_from_plex_guid("somethingelse://123") is None


def test_get_imdb_id_from_plex_guid_edge_cases():
    # Sin query string
    assert (
        plex_client.get_imdb_id_from_plex_guid(
            "com.plexapp.agents.imdb://tt0000001"
        )
        == "tt0000001"
    )
    # Cadena vacía
    assert plex_client.get_imdb_id_from_plex_guid("") is None
    # Texto que contiene imdb:// pero sin id después → None
    assert plex_client.get_imdb_id_from_plex_guid("xxx imdb://") is None


# -------------------------------------------------------------
# get_imdb_id_from_movie
# -------------------------------------------------------------


def test_get_imdb_id_from_movie_with_guids():
    g1 = SimpleNamespace(id="other://abc")
    g2 = SimpleNamespace(id="imdb://tt9999999")
    movie = SimpleNamespace(guids=[g1, g2], guid=None)
    assert plex_client.get_imdb_id_from_movie(movie) == "tt9999999"


def test_get_imdb_id_from_movie_fallback_to_guid():
    # Sin guids, pero con guid principal
    movie = SimpleNamespace(guids=None, guid="imdb://tt1111111")
    assert plex_client.get_imdb_id_from_movie(movie) == "tt1111111"


def test_get_imdb_id_from_movie_no_imdb_anywhere():
    g1 = SimpleNamespace(id="other://abc")
    movie = SimpleNamespace(guids=[g1], guid="other://xyz")
    assert plex_client.get_imdb_id_from_movie(movie) is None


# -------------------------------------------------------------
# get_best_search_title
# -------------------------------------------------------------


def test_get_best_search_title():
    movie = SimpleNamespace(originalTitle="Original Title", title="Fallback")
    assert plex_client.get_best_search_title(movie) == "Original Title"

    movie2 = SimpleNamespace(originalTitle="", title="Fallback 2")
    assert plex_client.get_best_search_title(movie2) == "Fallback 2"

    movie3 = SimpleNamespace()
    assert plex_client.get_best_search_title(movie3) == ""


def test_get_best_search_title_whitespace_and_non_string():
    # originalTitle solo espacios → se ignora, cae a title
    movie = SimpleNamespace(originalTitle="   ", title="Title OK")
    assert plex_client.get_best_search_title(movie) == "Title OK"

    # originalTitle no str, title sí str
    movie2 = SimpleNamespace(originalTitle=123, title="By Title")
    assert plex_client.get_best_search_title(movie2) == "By Title"

    # Ninguno válido → cadena vacía
    movie3 = SimpleNamespace(originalTitle=None, title=None)
    assert plex_client.get_best_search_title(movie3) == ""