from backend.reporting import open_all_csv_writer, open_filtered_csv_writer_only_if_rows


def test_open_all_csv_writer_creates_file(tmp_path):
    path = tmp_path / "report_all.csv"
    with open_all_csv_writer(str(path)) as writer:
        writer.write_row({"title": "Example", "decision": "KEEP"})

    assert path.exists()
    content = path.read_text(encoding="utf-8")
    header = content.splitlines()[0]
    assert "title" in header
    assert "decision" in header


def test_open_filtered_csv_writer_skips_empty(tmp_path):
    path = tmp_path / "report_filtered.csv"
    with open_filtered_csv_writer_only_if_rows(str(path)):
        pass

    assert not path.exists()
