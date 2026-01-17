import pandas as pd

from frontend.tabs import charts


def test_ordered_options_prefers_known_order_and_dedupes():
    values = ["KEEP", "DELETE", "MAYBE", "KEEP", None, ""]
    ordered = charts._ordered_options(values, ["KEEP", "MAYBE", "DELETE"])

    assert ordered == ["KEEP", "MAYBE", "DELETE"]


def test_mark_imdb_outliers_tags_extremes():
    df = pd.DataFrame(
        [
            {"imdb_rating": charts.IMDB_OUTLIER_HIGH},
            {"imdb_rating": charts.IMDB_OUTLIER_LOW},
            {"imdb_rating": (charts.IMDB_OUTLIER_HIGH + charts.IMDB_OUTLIER_LOW) / 2},
        ]
    )
    tagged = charts._mark_imdb_outliers(df)

    assert tagged.loc[0, "imdb_outlier"] == "Alta"
    assert tagged.loc[1, "imdb_outlier"] == "Baja"
    assert pd.isna(tagged.loc[2, "imdb_outlier"])


def test_movie_tooltips_include_expected_fields():
    df = pd.DataFrame(
        [
            {
                "title": "Movie",
                "year": 2001,
                "library": "Lib",
                "decision": "KEEP",
                "imdb_rating": 7.1,
                "rt_score": 90,
                "metacritic_score": 80,
                "imdb_votes": 1000,
                "file_size_gb": 1.25,
            }
        ]
    )
    tooltips = charts._movie_tooltips(df)
    tooltip_dicts = [tip.to_dict() for tip in tooltips]
    fields = {item.get("field") for item in tooltip_dicts}

    assert {
        "title",
        "year",
        "library",
        "decision",
        "imdb_rating",
        "rt_score",
        "metacritic_score",
        "imdb_votes",
        "file_size_gb",
    }.issubset(fields)
