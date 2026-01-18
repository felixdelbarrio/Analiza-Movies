from frontend.delete_logic import delete_files_from_rows


def test_delete_files_from_rows_dry_run(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("data", encoding="utf-8")

    ok, err, logs = delete_files_from_rows(
        [{"file": str(target), "title": "File"}], delete_dry_run=True
    )

    assert ok == 1
    assert err == 0
    assert target.exists()
    assert any("DRY" in line for line in logs)


def test_delete_files_from_rows_actual_delete(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("data", encoding="utf-8")

    ok, err, logs = delete_files_from_rows(
        [{"file": str(target), "title": "File"}], delete_dry_run=False
    )

    assert ok == 1
    assert err == 0
    assert not target.exists()
    assert any("deleted" in line for line in logs)
