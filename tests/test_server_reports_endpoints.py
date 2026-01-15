import pandas as pd
import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from server.api.app import create_app
from server.api.caching.file_cache import FileCache
from server.api.settings import Settings
import server.api.deps as deps
import server.api.routers.reports as reports_router


def _settings() -> Settings:
    return Settings(
        log_level="INFO",
        cors_origins_raw="*",
        cors_allow_credentials=False,
        gzip_min_size=0,
        file_cache_max_entries=4,
        file_cache_ttl_seconds=0.0,
        file_read_max_attempts=1,
        file_read_retry_sleep_s=0.0,
    )


def _write_csv(path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_reports_endpoints_with_files(monkeypatch, tmp_path):
    report_all = tmp_path / "report_all.csv"
    report_filtered = tmp_path / "report_filtered.csv"
    metadata_fix = tmp_path / "metadata_fix.csv"

    _write_csv(report_all, [{"title": "Movie A"}, {"title": "Movie B"}])
    _write_csv(report_filtered, [{"title": "Movie B"}])
    _write_csv(metadata_fix, [{"title": "Fix"}])

    monkeypatch.setattr(reports_router, "REPORT_ALL_PATH", report_all)
    monkeypatch.setattr(reports_router, "REPORT_FILTERED_PATH", report_filtered)
    monkeypatch.setattr(reports_router, "METADATA_FIX_PATH", metadata_fix)

    app = create_app()
    cache = FileCache(_settings())
    app.dependency_overrides[deps.get_file_cache] = lambda: cache

    client = TestClient(app)

    res_all = client.get("/reports/all?limit=10&offset=0")
    assert res_all.status_code == 200
    assert res_all.json()["total"] == 2

    res_filtered = client.get("/reports/filtered?limit=10&offset=0")
    assert res_filtered.status_code == 200
    assert res_filtered.json()["total"] == 1

    res_meta = client.get("/reports/metadata-fix?limit=10&offset=0")
    assert res_meta.status_code == 200
    assert res_meta.json()["total"] == 1


def test_reports_filtered_204_when_missing(monkeypatch, tmp_path):
    report_all = tmp_path / "report_all.csv"
    _write_csv(report_all, [{"title": "Movie A"}])

    monkeypatch.setattr(reports_router, "REPORT_ALL_PATH", report_all)
    monkeypatch.setattr(
        reports_router, "REPORT_FILTERED_PATH", tmp_path / "missing.csv"
    )
    monkeypatch.setattr(
        reports_router, "METADATA_FIX_PATH", tmp_path / "missing_meta.csv"
    )

    app = create_app()
    cache = FileCache(_settings())
    app.dependency_overrides[deps.get_file_cache] = lambda: cache

    client = TestClient(app)
    res_filtered = client.get("/reports/filtered?empty_as_204=true")

    assert res_filtered.status_code == 204
