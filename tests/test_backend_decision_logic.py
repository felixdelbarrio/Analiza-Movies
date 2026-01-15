from backend.decision_logic import sort_filtered_rows


def test_sort_filtered_rows_orders_by_decision_and_votes():
    rows = [
        {"decision": "KEEP", "imdb_votes": 100, "imdb_rating": 8.0, "title": "Keep A"},
        {
            "decision": "DELETE",
            "imdb_votes": 10,
            "imdb_rating": 1.0,
            "title": "Delete A",
        },
        {
            "decision": "MAYBE",
            "imdb_votes": 999,
            "imdb_rating": 5.0,
            "title": "Maybe A",
        },
        {
            "decision": "DELETE",
            "imdb_votes": 200,
            "imdb_rating": 1.0,
            "title": "Delete B",
        },
    ]

    sorted_rows = sort_filtered_rows(rows)
    titles = [r["title"] for r in sorted_rows]

    assert titles == ["Delete B", "Delete A", "Maybe A", "Keep A"]
