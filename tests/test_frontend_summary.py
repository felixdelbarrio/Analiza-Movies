import pandas as pd

from frontend.summary import compute_summary


def test_compute_summary_counts_and_sizes():
    df = pd.DataFrame(
        [
            {"decision": "KEEP", "file_size_gb": 1.0, "imdb_rating": 7.0},
            {"decision": "DELETE", "file_size_gb": 2.0, "imdb_rating": 5.0},
            {"decision": "MAYBE", "file_size_gb": 3.0, "imdb_rating": 6.0},
        ]
    )

    summary = compute_summary(df)

    assert summary["total_count"] == 3
    assert summary["keep_count"] == 1
    assert summary["delete_count"] == 1
    assert summary["maybe_count"] == 1
    assert summary["dm_count"] == 2
    assert summary["total_size_gb"] == 6.0
    assert summary["dm_size_gb"] == 5.0
