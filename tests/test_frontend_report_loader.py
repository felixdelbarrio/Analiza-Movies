import pandas as pd

from frontend.report_loader import load_reports


def test_load_reports_adds_derived_columns_and_strips_thumb(tmp_path):
    all_path = tmp_path / "report_all.csv"
    df_all = pd.DataFrame(
        [
            {
                "title": "Movie",
                "file_size": 1024,
                "omdb_json": "{}",
                "thumb": "http://thumb",
            }
        ]
    )
    df_all.to_csv(all_path, index=False)

    loaded_all, loaded_filtered = load_reports(
        str(all_path), str(tmp_path / "missing.csv")
    )

    assert loaded_filtered is None
    assert "file_size_gb" in loaded_all.columns
    assert "thumb" not in loaded_all.columns


def test_load_reports_reads_filtered_when_present(tmp_path):
    all_path = tmp_path / "report_all.csv"
    filtered_path = tmp_path / "report_filtered.csv"

    pd.DataFrame([{"title": "Movie", "file_size": 2048}]).to_csv(all_path, index=False)
    pd.DataFrame([{"title": "Movie", "file_size": 2048}]).to_csv(
        filtered_path, index=False
    )

    loaded_all, loaded_filtered = load_reports(str(all_path), str(filtered_path))

    assert loaded_filtered is not None
    assert len(loaded_all) == 1
    assert len(loaded_filtered) == 1
