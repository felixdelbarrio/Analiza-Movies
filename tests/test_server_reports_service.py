import pandas as pd

from server.api.services.reports import df_to_page, prepare_search_blob


def test_prepare_search_blob_and_df_to_page_query():
    df = pd.DataFrame(
        [
            {"title": "Best Movie", "imdb_id": "tt123", "file": "A"},
            {"title": "Other", "imdb_id": "tt999", "file": "B"},
        ]
    )
    prepare_search_blob(df)
    assert "__search_blob" in df.columns

    page = df_to_page(df, offset=0, limit=10, query="best")
    assert page["total"] == 1
    assert page["items"][0]["title"] == "Best Movie"
