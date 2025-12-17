from backend import omdb_client


# -------------------------------------------------------------
# normalize_imdb_votes
# -------------------------------------------------------------


def test_normalize_imdb_votes_variants():
    # Caso típico con comas
    assert omdb_client.normalize_imdb_votes("1,234") == 1234
    # Ya es numérico
    assert omdb_client.normalize_imdb_votes(5678) == 5678
    # Sin datos útiles
    assert omdb_client.normalize_imdb_votes("N/A") is None
    assert omdb_client.normalize_imdb_votes(None) is None


def test_normalize_imdb_votes_edge_cases():
    # Strings con espacios
    assert omdb_client.normalize_imdb_votes("  42 ") == 42
    # Combinaciones raras, pero parseables
    assert omdb_client.normalize_imdb_votes("0") == 0
    # No parseable → None
    assert omdb_client.normalize_imdb_votes("abc") is None
    assert omdb_client.normalize_imdb_votes({}) is None  # tipo extraño
    # Float → se castea a int
    assert omdb_client.normalize_imdb_votes(12.9) == 12


# -------------------------------------------------------------
# parse_rt_score_from_omdb / parse_imdb_rating_from_omdb / extract_year_from_omdb
# -------------------------------------------------------------


def test_parse_rt_and_imdb_rating_and_year():
    data = {
        "Ratings": [{"Source": "Rotten Tomatoes", "Value": "85%"}],
        "imdbRating": "7.3",
        "Year": "2019",
    }
    assert omdb_client.parse_rt_score_from_omdb(data) == 85
    assert abs(omdb_client.parse_imdb_rating_from_omdb(data) - 7.3) < 1e-6
    assert omdb_client.extract_year_from_omdb(data) == 2019


def test_parse_rt_score_from_omdb_missing_and_malformed():
    # Sin Rotten Tomatoes
    data_no_rt = {
        "Ratings": [
            {"Source": "Metacritic", "Value": "60/100"},
            {"Source": "Internet Movie Database", "Value": "7.0/10"},
        ]
    }
    assert omdb_client.parse_rt_score_from_omdb(data_no_rt) is None

    # Valor malformado
    data_bad_value = {
        "Ratings": [{"Source": "Rotten Tomatoes", "Value": "bad"}],
    }
    assert omdb_client.parse_rt_score_from_omdb(data_bad_value) is None

    # Ratings vacío o faltante
    assert omdb_client.parse_rt_score_from_omdb({"Ratings": []}) is None
    assert omdb_client.parse_rt_score_from_omdb({}) is None


def test_parse_imdb_rating_from_omdb_edge_cases():
    assert omdb_client.parse_imdb_rating_from_omdb({"imdbRating": "N/A"}) is None
    assert omdb_client.parse_imdb_rating_from_omdb({}) is None
    assert omdb_client.parse_imdb_rating_from_omdb({"imdbRating": None}) is None

    # Numérico directo
    assert abs(omdb_client.parse_imdb_rating_from_omdb({"imdbRating": 8.1}) - 8.1) < 1e-6

    # Malformado → None
    assert omdb_client.parse_imdb_rating_from_omdb({"imdbRating": "bad"}) is None


def test_extract_year_from_omdb_various_formats():
    # Año simple
    assert omdb_client.extract_year_from_omdb({"Year": "2010"}) == 2010
    # Rango de años típico de series/películas
    assert omdb_client.extract_year_from_omdb({"Year": "2010–2012"}) == 2010
    assert omdb_client.extract_year_from_omdb({"Year": "2010–"}) == 2010
    # Con espacios
    assert omdb_client.extract_year_from_omdb({"Year": " 1999 "}) == 1999
    # N/A, vacío o no convertible
    assert omdb_client.extract_year_from_omdb({"Year": "N/A"}) is None
    assert omdb_client.extract_year_from_omdb({"Year": ""}) is None
    assert omdb_client.extract_year_from_omdb({}) is None


# -------------------------------------------------------------
# extract_ratings_from_omdb + is_omdb_data_empty_for_ratings
# -------------------------------------------------------------


def test_extract_ratings_and_empty():
    data = {
        "Ratings": [{"Source": "Rotten Tomatoes", "Value": "90%"}],
        "imdbRating": "8.0",
        "imdbVotes": "10,000",
    }
    r, v, rt = omdb_client.extract_ratings_from_omdb(data)
    assert abs(r - 8.0) < 1e-6
    assert v == 10000
    assert rt == 90

    # Casos claramente vacíos
    assert omdb_client.is_omdb_data_empty_for_ratings({}) is True
    assert omdb_client.is_omdb_data_empty_for_ratings(None) is True


def test_extract_ratings_partial_information():
    # Solo IMDb rating
    data_rating_only = {"imdbRating": "7.5"}
    r, v, rt = omdb_client.extract_ratings_from_omdb(data_rating_only)
    assert abs(r - 7.5) < 1e-6
    assert v is None
    assert rt is None

    # Solo votos
    data_votes_only = {"imdbVotes": "12,345"}
    r, v, rt = omdb_client.extract_ratings_from_omdb(data_votes_only)
    assert r is None
    assert v == 12345
    assert rt is None

    # Solo Rotten
    data_rt_only = {
        "Ratings": [{"Source": "Rotten Tomatoes", "Value": "55%"}],
    }
    r, v, rt = omdb_client.extract_ratings_from_omdb(data_rt_only)
    assert r is None
    assert v is None
    assert rt == 55


def test_is_omdb_data_empty_for_ratings_false_when_any_present():
    # IMDb rating presente
    assert (
        omdb_client.is_omdb_data_empty_for_ratings({"imdbRating": "6.0"})
        is False
    )
    # imdbVotes presente
    assert (
        omdb_client.is_omdb_data_empty_for_ratings({"imdbVotes": "1,000"})
        is False
    )
    # Rotten Tomatoes presente
    assert (
        omdb_client.is_omdb_data_empty_for_ratings(
            {"Ratings": [{"Source": "Rotten Tomatoes", "Value": "10%"}]}
        )
        is False
    )

    # Todos "vacíos" → debe ser True
    assert (
        omdb_client.is_omdb_data_empty_for_ratings(
            {
                "imdbRating": "N/A",
                "imdbVotes": "N/A",
                "Ratings": [],
            }
        )
        is True
    )