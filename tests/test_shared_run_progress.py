from __future__ import annotations

from pathlib import Path

from shared.run_progress import load_progress, write_progress


def test_write_progress_falls_back_when_atomic_replace_loses_directory(
    monkeypatch, tmp_path
):
    progress_path = tmp_path / "profiles" / "plex-a" / "last_run.progress.json"
    payload = {"status": "running", "profile_id": "plex-a"}

    original_replace = Path.replace

    def _always_fail_replace(self: Path, target: Path) -> Path:
        raise FileNotFoundError(2, "No such file or directory", str(self))

    monkeypatch.setattr(Path, "replace", _always_fail_replace)

    result = write_progress(progress_path, payload)

    monkeypatch.setattr(Path, "replace", original_replace)

    assert result == payload
    assert progress_path.exists()
    assert load_progress(progress_path) == payload
