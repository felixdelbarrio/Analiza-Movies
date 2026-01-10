import pandas as pd

from frontend.front_stats import compute_global_imdb_mean_from_df, compute_global_imdb_mean_from_report_all


def test_compute_global_imdb_mean_from_df():
    df = pd.DataFrame([{"imdb_rating": 7.0}, {"imdb_rating": "8.0"}, {"imdb_rating": None}])
    assert compute_global_imdb_mean_from_df(df) == 7.5


def test_compute_global_imdb_mean_from_report_all(tmp_path):
    path = tmp_path / "report_all.csv"
    pd.DataFrame([{"imdb_rating": 6.0}, {"imdb_rating": 8.0}]).to_csv(path, index=False)

    assert compute_global_imdb_mean_from_report_all(path) == 7.0
