from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

import server.api.services.file_actions as file_actions_service
from server.api.app import create_app


def _write_report(path, rows) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_delete_action_only_allows_files_present_in_reports(
    monkeypatch, tmp_path
) -> None:
    allowed_media = tmp_path / "movie-ok.mkv"
    allowed_media.write_text("ok", encoding="utf-8")
    report_filtered = tmp_path / "report_filtered.csv"
    report_all = tmp_path / "report_all.csv"
    _write_report(report_filtered, [{"title": "Movie OK", "file": str(allowed_media)}])
    _write_report(report_all, [{"title": "Movie OK", "file": str(allowed_media)}])

    monkeypatch.setattr(
        file_actions_service,
        "get_report_filtered_path",
        lambda profile_id=None: report_filtered,
    )
    monkeypatch.setattr(
        file_actions_service,
        "get_report_all_path",
        lambda profile_id=None: report_all,
    )

    client = TestClient(create_app())
    response = client.post(
        "/actions/delete",
        json={
            "profile_id": "plex-test",
            "dry_run": True,
            "rows": [{"title": "Movie OK", "file": str(allowed_media)}],
        },
    )

    assert response.status_code == 200
    assert response.json()["ok"] == 1
    assert response.json()["err"] == 0

    forbidden = tmp_path / "secret.txt"
    forbidden.write_text("secret", encoding="utf-8")
    denied = client.post(
        "/actions/delete",
        json={
            "profile_id": "plex-test",
            "dry_run": True,
            "rows": [{"title": "Secret", "file": str(forbidden)}],
        },
    )

    assert denied.status_code == 200
    assert denied.json()["ok"] == 0
    assert denied.json()["err"] == 1
    assert "ruta no autorizada" in "\n".join(denied.json()["logs"])


def test_delete_action_removes_authorized_file(monkeypatch, tmp_path) -> None:
    media_path = tmp_path / "movie-delete.mkv"
    media_path.write_text("delete-me", encoding="utf-8")
    report_filtered = tmp_path / "report_filtered.csv"
    _write_report(report_filtered, [{"title": "Movie Delete", "file": str(media_path)}])

    monkeypatch.setattr(
        file_actions_service,
        "get_report_filtered_path",
        lambda profile_id=None: report_filtered,
    )
    monkeypatch.setattr(
        file_actions_service,
        "get_report_all_path",
        lambda profile_id=None: tmp_path / "missing-report-all.csv",
    )

    client = TestClient(create_app())
    response = client.post(
        "/actions/delete",
        json={
            "profile_id": "plex-test",
            "dry_run": False,
            "rows": [{"title": "Movie Delete", "file": str(media_path)}],
        },
    )

    assert response.status_code == 200
    assert response.json()["ok"] == 1
    assert response.json()["err"] == 0
    assert not media_path.exists()
