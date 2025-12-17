import math

import pandas as pd

from backend import stats


# ---------------------------------------------------------
# compute_global_imdb_mean_from_df
# ---------------------------------------------------------


def test_compute_global_imdb_mean_from_df():
    df = pd.DataFrame({"imdb_rating": [7.0, 8.0, None, "9.0"]})
    mean = stats.compute_global_imdb_mean_from_df(df)
    # (7 + 8 + 9) / 3 = 8.0
    assert isinstance(mean, float)
    assert abs(mean - 8.0) < 1e-6


def test_compute_global_imdb_mean_from_df_no_column():
    df = pd.DataFrame({"other": [1, 2, 3]})
    mean = stats.compute_global_imdb_mean_from_df(df)
    assert mean is None


def test_compute_global_imdb_mean_from_df_all_nan():
    df = pd.DataFrame({"imdb_rating": [None, "bad", float("nan")]})
    mean = stats.compute_global_imdb_mean_from_df(df)
    assert mean is None


# ---------------------------------------------------------
# get_global_imdb_mean_from_cache
# ---------------------------------------------------------


def test_global_imdb_mean_is_number_from_cache_fallback_default(monkeypatch):
    # omdb_cache vacío → debe usar BAYES_GLOBAL_MEAN_DEFAULT
    monkeypatch.setattr(stats, "omdb_cache", {})
    # Reset de caché interna
    monkeypatch.setattr(stats, "_GLOBAL_IMDB_MEAN_FROM_CACHE", None)
    monkeypatch.setattr(stats, "_GLOBAL_IMDB_MEAN_SOURCE", None)
    monkeypatch.setattr(stats, "_GLOBAL_IMDB_MEAN_COUNT", None)

    mean = stats.get_global_imdb_mean_from_cache()
    assert isinstance(mean, float)
    assert not math.isnan(mean)
    assert mean == stats.BAYES_GLOBAL_MEAN_DEFAULT

    m2, source, count = stats.get_global_imdb_mean_info()
    assert m2 == mean
    assert "default" in source
    assert isinstance(count, int)


def test_global_imdb_mean_from_cache_with_data(monkeypatch):
    # Preparamos un omdb_cache pequeño con ratings
    omdb_cache = {
        "k1": {"imdbRating": "7.0"},
        "k2": {"imdbRating": "9.0"},
        "k3": {"imdbRating": "N/A"},  # se ignora
    }
    monkeypatch.setattr(stats, "omdb_cache", omdb_cache)
    # Bajamos el mínimo de títulos para que use la media calculada
    monkeypatch.setattr(stats, "BAYES_MIN_TITLES_FOR_GLOBAL_MEAN", 2, raising=False)

    # Reset de caché interna
    monkeypatch.setattr(stats, "_GLOBAL_IMDB_MEAN_FROM_CACHE", None)
    monkeypatch.setattr(stats, "_GLOBAL_IMDB_MEAN_SOURCE", None)
    monkeypatch.setattr(stats, "_GLOBAL_IMDB_MEAN_COUNT", None)

    mean = stats.get_global_imdb_mean_from_cache()
    # media de 7.0 y 9.0 = 8.0
    assert abs(mean - 8.0) < 1e-6

    m2, source, count = stats.get_global_imdb_mean_info()
    assert abs(m2 - 8.0) < 1e-6
    assert "omdb_cache" in source
    assert count == 2


# ---------------------------------------------------------
# AUTO-UMBRAL KEEP / DELETE (global)
# ---------------------------------------------------------


def test_get_auto_keep_rating_threshold_uses_percentile(monkeypatch):
    # omdb_cache con tres ratings IMDb
    omdb_cache = {
        "k1": {"imdbRating": "5.0"},
        "k2": {"imdbRating": "6.0"},
        "k3": {"imdbRating": "9.0"},
    }
    monkeypatch.setattr(stats, "omdb_cache", omdb_cache)

    # Configuramos percentil y mínimo
    monkeypatch.setattr(stats, "RATING_MIN_TITLES_FOR_AUTO", 3, raising=False)
    monkeypatch.setattr(stats, "AUTO_KEEP_RATING_PERCENTILE", 0.5, raising=False)
    monkeypatch.setattr(stats, "IMDB_KEEP_MIN_RATING", 7.5, raising=False)

    # Reset cachés internas
    monkeypatch.setattr(stats, "_RATINGS_LIST", None)
    monkeypatch.setattr(stats, "_RATINGS_COUNT", 0)
    monkeypatch.setattr(stats, "_AUTO_KEEP_RATING_THRESHOLD", None)

    thr = stats.get_auto_keep_rating_threshold()
    # ratings ordenados: [5.0, 6.0, 9.0]
    # p=0.5 → idx = int(0.5*(3-1)) = 1 → 6.0
    assert abs(thr - 6.0) < 1e-6


def test_get_auto_keep_rating_threshold_fallback(monkeypatch):
    # Solo un rating → no llega al mínimo para auto
    omdb_cache = {"k1": {"imdbRating": "6.0"}}
    monkeypatch.setattr(stats, "omdb_cache", omdb_cache)

    monkeypatch.setattr(stats, "RATING_MIN_TITLES_FOR_AUTO", 3, raising=False)
    monkeypatch.setattr(stats, "IMDB_KEEP_MIN_RATING", 7.5, raising=False)

    monkeypatch.setattr(stats, "_RATINGS_LIST", None)
    monkeypatch.setattr(stats, "_RATINGS_COUNT", 0)
    monkeypatch.setattr(stats, "_AUTO_KEEP_RATING_THRESHOLD", None)

    thr = stats.get_auto_keep_rating_threshold()
    assert thr == 7.5  # fallback IMDB_KEEP_MIN_RATING


def test_get_auto_delete_rating_threshold_uses_percentile(monkeypatch):
    omdb_cache = {
        "k1": {"imdbRating": "3.0"},
        "k2": {"imdbRating": "4.0"},
        "k3": {"imdbRating": "9.0"},
    }
    monkeypatch.setattr(stats, "omdb_cache", omdb_cache)

    monkeypatch.setattr(stats, "RATING_MIN_TITLES_FOR_AUTO", 3, raising=False)
    monkeypatch.setattr(stats, "AUTO_DELETE_RATING_PERCENTILE", 0.1, raising=False)
    monkeypatch.setattr(stats, "IMDB_DELETE_MAX_RATING", 5.0, raising=False)

    monkeypatch.setattr(stats, "_RATINGS_LIST", None)
    monkeypatch.setattr(stats, "_RATINGS_COUNT", 0)
    monkeypatch.setattr(stats, "_AUTO_DELETE_RATING_THRESHOLD", None)

    thr = stats.get_auto_delete_rating_threshold()
    # ratings ordenados: [3.0, 4.0, 9.0]
    # p=0.1 → idx=int(0.1*(3-1))=0 → 3.0
    assert abs(thr - 3.0) < 1e-6


def test_get_auto_delete_rating_threshold_fallback(monkeypatch):
    omdb_cache = {"k1": {"imdbRating": "4.0"}}
    monkeypatch.setattr(stats, "omdb_cache", omdb_cache)

    monkeypatch.setattr(stats, "RATING_MIN_TITLES_FOR_AUTO", 3, raising=False)
    monkeypatch.setattr(stats, "IMDB_DELETE_MAX_RATING", 5.5, raising=False)

    monkeypatch.setattr(stats, "_RATINGS_LIST", None)
    monkeypatch.setattr(stats, "_RATINGS_COUNT", 0)
    monkeypatch.setattr(stats, "_AUTO_DELETE_RATING_THRESHOLD", None)

    thr = stats.get_auto_delete_rating_threshold()
    assert thr == 5.5  # fallback IMDB_DELETE_MAX_RATING


# ---------------------------------------------------------
# AUTO-UMBRAL KEEP / DELETE (NO_RT)
# ---------------------------------------------------------


def test_get_auto_keep_rating_threshold_no_rt(monkeypatch):
    # k1, k2 sin RT, k3 con RT (debe ser ignorada)
    omdb_cache = {
        "k1": {"imdbRating": "6.0"},
        "k2": {"imdbRating": "8.0"},
        "k3": {
            "imdbRating": "9.0",
            "Ratings": [{"Source": "Rotten Tomatoes", "Value": "90%"}],
        },
    }
    monkeypatch.setattr(stats, "omdb_cache", omdb_cache)

    monkeypatch.setattr(stats, "RATING_MIN_TITLES_FOR_AUTO", 2, raising=False)
    monkeypatch.setattr(stats, "AUTO_KEEP_RATING_PERCENTILE", 0.5, raising=False)

    # Reset cachés internas específicas NO_RT
    monkeypatch.setattr(stats, "_RATINGS_NO_RT_LIST", None)
    monkeypatch.setattr(stats, "_RATINGS_NO_RT_COUNT", 0)
    monkeypatch.setattr(stats, "_AUTO_KEEP_RATING_THRESHOLD_NO_RT", None)

    thr = stats.get_auto_keep_rating_threshold_no_rt()
    # ratings sin RT: [6.0, 8.0] → ordenados
    # p=0.5 → idx=int(0.5*(2-1))=0 → 6.0
    assert abs(thr - 6.0) < 1e-6


def test_get_auto_keep_rating_threshold_no_rt_fallback(monkeypatch):
    # Solo un título sin RT → no llega al mínimo
    omdb_cache = {
        "k1": {"imdbRating": "7.0"},
        "k2": {
            "imdbRating": "8.0",
            "Ratings": [{"Source": "Rotten Tomatoes", "Value": "90%"}],
        },
    }
    monkeypatch.setattr(stats, "omdb_cache", omdb_cache)

    # Mínimo 2 para auto, solo 1 sin RT
    monkeypatch.setattr(stats, "RATING_MIN_TITLES_FOR_AUTO", 2, raising=False)

    # Forzamos un valor conocido para el umbral global
    monkeypatch.setattr(stats, "_AUTO_KEEP_RATING_THRESHOLD", 7.5)

    monkeypatch.setattr(stats, "_RATINGS_NO_RT_LIST", None)
    monkeypatch.setattr(stats, "_RATINGS_NO_RT_COUNT", 0)
    monkeypatch.setattr(stats, "_AUTO_KEEP_RATING_THRESHOLD_NO_RT", None)

    thr = stats.get_auto_keep_rating_threshold_no_rt()
    # Debe reutilizar el umbral global KEEP
    assert thr == 7.5


def test_get_auto_delete_rating_threshold_no_rt(monkeypatch):
    # k1, k2 sin RT, k3 con RT
    omdb_cache = {
        "k1": {"imdbRating": "3.0"},
        "k2": {"imdbRating": "5.0"},
        "k3": {
            "imdbRating": "9.0",
            "Ratings": [{"Source": "Rotten Tomatoes", "Value": "90%"}],
        },
    }
    monkeypatch.setattr(stats, "omdb_cache", omdb_cache)

    monkeypatch.setattr(stats, "RATING_MIN_TITLES_FOR_AUTO", 2, raising=False)
    monkeypatch.setattr(stats, "AUTO_DELETE_RATING_PERCENTILE", 0.0, raising=False)

    monkeypatch.setattr(stats, "_RATINGS_NO_RT_LIST", None)
    monkeypatch.setattr(stats, "_RATINGS_NO_RT_COUNT", 0)
    monkeypatch.setattr(stats, "_AUTO_DELETE_RATING_THRESHOLD_NO_RT", None)

    thr = stats.get_auto_delete_rating_threshold_no_rt()
    # ratings sin RT: [3.0, 5.0] → p=0.0 → 3.0
    assert abs(thr - 3.0) < 1e-6


def test_get_auto_delete_rating_threshold_no_rt_fallback(monkeypatch):
    omdb_cache = {
        "k1": {"imdbRating": "4.0"},
        "k2": {
            "imdbRating": "6.0",
            "Ratings": [{"Source": "Rotten Tomatoes", "Value": "80%"}],
        },
    }
    monkeypatch.setattr(stats, "omdb_cache", omdb_cache)

    monkeypatch.setattr(stats, "RATING_MIN_TITLES_FOR_AUTO", 2, raising=False)

    # Forzamos un valor conocido para el umbral global DELETE
    monkeypatch.setattr(stats, "_AUTO_DELETE_RATING_THRESHOLD", 4.5)

    monkeypatch.setattr(stats, "_RATINGS_NO_RT_LIST", None)
    monkeypatch.setattr(stats, "_RATINGS_NO_RT_COUNT", 0)
    monkeypatch.setattr(stats, "_AUTO_DELETE_RATING_THRESHOLD_NO_RT", None)

    thr = stats.get_auto_delete_rating_threshold_no_rt()
    assert thr == 4.5