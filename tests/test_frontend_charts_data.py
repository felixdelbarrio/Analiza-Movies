import pandas as pd

from frontend.tabs import charts_data


def test_genres_agg_counts_by_genre():
    df = pd.DataFrame(
        [
            {
                "title": "Movie A",
                "decision": "KEEP",
                "omdb_json": '{"Genre": "Action, Drama"}',
            },
            {
                "title": "Movie B",
                "decision": "DELETE",
                "omdb_json": '{"Genre": "Action, Drama"}',
            },
        ]
    )

    agg = charts_data._genres_agg(df)
    assert not agg.empty

    pivot = (
        agg.pivot_table(index="genre", columns="decision", values="count", fill_value=0)
        .reset_index()
        .set_index("genre")
    )

    assert pivot.loc["Action", "KEEP"] == 1
    assert pivot.loc["Action", "DELETE"] == 1
    assert pivot.loc["Drama", "KEEP"] == 1
    assert pivot.loc["Drama", "DELETE"] == 1
